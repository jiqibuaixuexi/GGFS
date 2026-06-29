"""
GGFS — End-to-End Inference Pipeline (CNF ODE Solver + Best-of-N Filtering)
=============================================================================
Pathology-aware asymmetric routing, anatomy-aware morphological energy-guided
ODE integration, and Preference-Aligned Best-of-N selection for controllable
syndesmophyte editing in AS (Ankylosing Spondylitis) lumbar CT images.

Implements GGFS pipeline stages:
  (B) Pathological Routing — two-step cyclic pipeline for erasure transitions
  (C) CNF ODE Solver — time-truncated energy-guided velocity field integration
  (D) Best-of-N Filtering — ResNet-34 evaluator selects the top candidate

Requires:
    models.inpainting_flow_model_DAG  — InpaintingFlowUNet (CNF velocity predictor)
    models.SegModel                   — ResUNet34 (anatomy-aware energy critic)
    models.RewardModel                — PathologyRewardModel (ResNet-34 evaluator)
    utils.sampling                    — sample_energy_guided_cfg_editing
    utils.tools                       — extract_single_side_mask
"""

import os
import glob
import json
import math
import random
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.inpainting_flow_model_DAG import InpaintingFlowUNet
from models.SegModel import ResUNet34
from models.RewardModel import PathologyRewardModel
from utils.sampling import sample_energy_guided_cfg_editing
from utils.tools import extract_single_side_mask


# ==============================================================================
# Global Hyperparameters
# ==============================================================================

GOLDEN_PARAMS = {
    "blur_kernel_size": 41,
    "blur_sigma": 10.0,
    "plateau_factor": 1.0,
    "shift_pixels": 5,
    "energy_scale": 50.0,
    "guidance_start": 0.1,
    "guidance_end": 0.4,
    "gap_scale": 0.1,
}


# ==============================================================================
# Utilities
# ==============================================================================

def seed_everything(seed):
    """Lock all randomness sources for deterministic reproduction."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Stage B + C: Pathological Routing + CNF ODE Solver
# ==============================================================================

def generate_n_variants(
    flow_model,
    seg_model,
    orig_tensor_11,
    mask_tensor,
    s_score,
    t_score,
    side,
    num_variants=8,
    device="cuda",
    first_stage_score_scale=2.0,
    score_scale=1.5,
    golden_params=None,
):
    """
    GGFS Pathological Routing (Stage B): generate N candidate variants for a
    single image given a source→target editing instruction.

    Routing logic (asymmetric):
      - Erasure transition (any score component decreases):
          Two-step cyclic pipeline: Abnormal → Normal (score 0) → Target.
      - Pure proliferation (all components increase or stay):
          Direct one-step generation preserving anatomical background.

    Args:
        flow_model: Trained InpaintingFlowUNet (CNF velocity predictor)
        seg_model:  Frozen ResUNet34 (anatomy-aware energy critic)
        orig_tensor_11:  Source image [1, 3, H, W] normalized to [-1, 1]
        mask_tensor:     Binary mask [1, 1, H, W] (1 = editing region)
        s_score:         Source syndesmophyte grade (int, 0–4)
        t_score:         Target syndesmophyte grade (int, 0–4)
        side:            0 = anterior, 1 = posterior
        num_variants:    Number of candidates to generate (default 8)
        device:          Device string
        first_stage_score_scale: CFG score scale for the zeroing stage
        score_scale:     CFG score scale for the rebuild / single-stage step
        golden_params:   Dict of hyperparams forwarded to the ODE sampler

    Returns:
        variants_batch: [N, 3, H, W] tensor in [0, 1]
    """
    if golden_params is None:
        golden_params = GOLDEN_PARAMS

    s_class = torch.tensor([s_score], dtype=torch.long, device=device)
    t_class = torch.tensor([t_score], dtype=torch.long, device=device)
    side_tensor = torch.tensor([side], dtype=torch.long, device=device)
    zero_class = torch.zeros_like(t_class)

    # Determine routing: check for any score-component decrease
    composition_map = {
        0: [1.0, 0.0, 0.0, 0.0],
        1: [1.0, 1.0, 0.0, 0.0],
        2: [1.0, 0.0, 1.0, 0.0],
        3: [1.0, 1.0, 1.0, 0.0],
        4: [1.0, 1.0, 1.0, 1.0],
    }
    delta = [composition_map[t_score][i] - composition_map[s_score][i] for i in range(4)]
    has_erasure = any(d < 0 for d in delta)

    generated_variants = []

    for i in range(num_variants):
        current_seed = random.randint(10000, 99999)
        seed_everything(current_seed)

        current_brightness = random.uniform(0.75, 0.98)
        current_dimness = random.uniform(0.2, 0.5)

        # ================================================================
        # Branch A: Erasure needed → Two-stage
        # ================================================================
        if has_erasure:
            # Stage 1: Force to zero
            stage1_img_01 = sample_energy_guided_cfg_editing(
                flow_model=flow_model,
                seg_model=seg_model,
                real_img=orig_tensor_11,
                mask_cond=mask_tensor,
                source_score=s_class,
                target_score=zero_class,
                side_class=side_tensor,
                mask_scale=1.0,
                score_scale=first_stage_score_scale,
                steps=50,
                device=device,
                **golden_params,
            )

            if t_score == 0:
                final_img_01 = stage1_img_01
            else:
                # Stage 2: Rebuild target on clean canvas
                stage1_img_11 = stage1_img_01 * 2.0 - 1.0
                final_img_01 = sample_energy_guided_cfg_editing(
                    flow_model=flow_model,
                    seg_model=seg_model,
                    real_img=stage1_img_11,
                    mask_cond=mask_tensor,
                    source_score=zero_class,
                    target_score=t_class,
                    side_class=side_tensor,
                    mask_scale=1.0,
                    score_scale=score_scale,
                    steps=50,
                    device=device,
                    **golden_params,
                )

        # ================================================================
        # Branch B: Pure upgrade → Single-stage
        # ================================================================
        else:
            step_params = golden_params.copy()
            final_img_01 = sample_energy_guided_cfg_editing(
                flow_model=flow_model,
                seg_model=seg_model,
                real_img=orig_tensor_11,
                mask_cond=mask_tensor,
                source_score=s_class,
                target_score=t_class,
                side_class=side_tensor,
                mask_scale=1.0,
                score_scale=score_scale,
                steps=50,
                device=device,
                **step_params,
            )

        generated_variants.append(final_img_01)

    variants_batch = torch.cat(generated_variants, dim=0)  # [N, 3, H, W]
    return variants_batch


# ==============================================================================
# Stage D: Preference-Aligned Best-of-N Filtering
# ==============================================================================

def score_variants(reward_model, variants_batch, s_score, t_score, side, device="cuda"):
    """
    Batch-score N generated variants using the trained reward model.

    Args:
        reward_model:   Trained PathologyRewardModel
        variants_batch: [N, 3, 224, 224] tensor in [0, 1]
        s_score:        Source syndesmophyte score
        t_score:        Target syndesmophyte score
        side:           0 = anterior, 1 = posterior
        device:         Device string

    Returns:
        scores_list: list of float scores (length N)
    """
    N = variants_batch.size(0)

    # Replicate conditioning to match batch size
    s_tensor = torch.full((N,), s_score, dtype=torch.long, device=device)
    t_tensor = torch.full((N,), t_score, dtype=torch.long, device=device)
    side_tensor = torch.full((N,), side, dtype=torch.long, device=device)

    # ImageNet normalization (critical: match ResNet34 training distribution)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    normalized_batch = (variants_batch - mean) / std

    with torch.no_grad():
        scores = reward_model(normalized_batch, s_tensor, t_tensor, side_tensor)

    scores_list = scores.cpu().numpy().tolist()
    return scores_list


# ==============================================================================
# Stage D (cont.): End-to-End Best-of-N Pipeline Orchestrator
# ==============================================================================

def run_medical_editing_pipeline(
    flow_model, seg_model, reward_model,
    orig_tensor_11, mask_tensor,
    s_score, t_score, side,
    best_threshold,
    num_variants=8, device="cuda",
):
    """
    GGFS Best-of-N Filtering (Stage D): end-to-end single-image editing pipeline.

    1. Generate N candidates via pathological routing (Stage B + C).
    2. Upsample to 224×224 for ResNet-34 evaluator input.
    3. Score all candidates with the Preference-Aligned reward model.
    4. Select the highest-scoring variant unconditionally (pure Best-of-N).

    Args:
        flow_model, seg_model, reward_model: Pre-loaded GGFS models
        orig_tensor_11: Source image [1, 3, H, W] in [-1, 1]
        mask_tensor:    Binary mask [1, 1, H, W]
        s_score, t_score: Source and target syndesmophyte grades
        side:           0 = anterior, 1 = posterior
        best_threshold: Placeholder (kept for API compatibility)
        num_variants:   Number of candidates per image (default 8)
        device:         Device string

    Returns:
        (best_image_tensor, best_score, True)
          best_image_tensor: [1, 3, H, W] in [0, 1]
    """
    # 1. Generate N variants
    variants_batch = generate_n_variants(
        flow_model, seg_model, orig_tensor_11, mask_tensor,
        s_score, t_score, side,
        num_variants=num_variants, device=device,
    )

    # 2. Size alignment (Flow 128 → Reward 224)
    if variants_batch.shape[-1] != 224:
        eval_batch = F.interpolate(variants_batch, size=(224, 224), mode='bilinear', align_corners=False)
    else:
        eval_batch = variants_batch

    # 3. Score all variants
    scores = score_variants(reward_model, eval_batch, s_score, t_score, side, device=device)

    # 4. Best-of-N: pick the highest-scoring variant
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_image_tensor = variants_batch[best_idx:best_idx + 1]

    return best_image_tensor, best_score, True


# ==============================================================================
# HEM Augmentation: Automated Offline Database Construction (3-Fold CV)
# ==============================================================================

def load_models_for_fold(target_test_fold):
    """
    Load Flow, Seg, and Reward models trained on the complementary two folds.

    For 3-fold cross-validation, a model intended for testing on Fold K must be
    trained on the other two folds. This function loads the correct checkpoint
    triplet based on the target test fold.

    Args:
        target_test_fold: int (1, 2, or 3) — the fold to generate data FOR

    Returns:
        (flow_model, seg_model, reward_model, filter_threshold)
    """
    print(f"\nLoading models for target test fold {target_test_fold} ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    flow_model = InpaintingFlowUNet(img_ch=3, masked_img_ch=3, mask_ch=1, base_ch=128).to(device)
    seg_model = ResUNet34(n_classes=1, pretrained=False).to(device)
    reward_model = PathologyRewardModel(pretrained=False).to(device)

    if target_test_fold == 1:
        flow_ckpt = r"checkpoints\inpainting_flow_mse\train23_val1\inpainting_flow_epoch_50.pt"
        seg_ckpt = r"checkpoints\seg_model_checkpoints\t23_v1\best_resunet_34.pth"
        reward_ckpt = r"checkpoints\reward_model_checkpoints\filter_fold3_best.pth"
    elif target_test_fold == 2:
        flow_ckpt = r"checkpoints\inpainting_flow_mse\train13_val2\inpainting_flow_epoch_50.pt"
        seg_ckpt = r"checkpoints\seg_model_checkpoints\t13_v2\best_resunet_34.pth"
        reward_ckpt = r"checkpoints\reward_model_checkpoints\filter_fold2_best.pth"
    else:
        flow_ckpt = r"checkpoints\inpainting_flow_mse\train12_val3\inpainting_flow_epoch_50.pt"
        seg_ckpt = r"checkpoints\seg_model_checkpoints\t12_v3\best_resunet_34.pth"
        reward_ckpt = r"checkpoints\reward_model_checkpoints\filter_fold1_best.pth"

    flow_model.load_state_dict(torch.load(flow_ckpt, map_location=device))
    print("Flow model loaded.")
    seg_model.load_state_dict(torch.load(seg_ckpt, map_location=device))
    print("Seg model loaded.")

    checkpoint = torch.load(reward_ckpt, map_location=device)
    reward_model.load_state_dict(checkpoint['model_state_dict'])
    filter_threshold = checkpoint.get('best_threshold', 0.5)
    print(f"Reward model loaded (threshold: {filter_threshold:.4f}).")

    flow_model.eval()
    seg_model.eval()
    reward_model.eval()

    return flow_model, seg_model, reward_model, filter_threshold


def build_offline_database(quota_per_side=1000):
    """
    Automated production of synthetic CT images for all three cross-validation folds.

    For each fold K (1, 2, 3), class C (0–4), and side S (anterior/posterior):
      - Load models trained on complementary folds.
      - Shuffle and draw from the valid source-image pool WITHOUT replacement.
      - Generate quota_per_side images using the Best-of-N pipeline.
      - Save with UUID-based unique filenames.

    Total output: 3 folds × 5 classes × 2 sides × quota_per_side images.

    Args:
        quota_per_side: Target number of images per (fold, class, side) combination.
    """
    output_base_dir = "offline_database"
    transform = transforms.Compose([transforms.ToTensor()])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_dirs = {
        1: (r"data\cv_ds\t12_v3\val_bbox",
            r"data\cv_ds\t12_v3\val_point_based_masks",
            r"data\cv_ds\t12_v3\val_labels_inpainting.csv"),
        2: (r"data\cv_ds\t13_v2\val_bbox",
            r"data\cv_ds\t13_v2\val_point_based_masks",
            r"data\cv_ds\t13_v2\val_labels_inpainting.csv"),
        3: (r"data\cv_ds\t23_v1\val_bbox",
            r"data\cv_ds\t23_v1\val_point_based_masks",
            r"data\cv_ds\t23_v1\val_labels_inpainting.csv"),
    }

    for target_test_fold in [1, 2, 3]:
        print("\n" + "=" * 60)
        print(f" Database {target_test_fold} (for testing Fold {target_test_fold})")
        print("=" * 60)

        # 1. Load strictly isolated models
        flow_model, seg_model, reward_model, filter_threshold = load_models_for_fold(target_test_fold)

        # 2. Collect valid source images from complementary folds with ground-truth labels
        valid_base_images = []
        valid_mask_paths = {}
        valid_scores = {}

        train_folds = [f for f in [1, 2, 3] if f != target_test_fold]
        for tf in train_folds:
            img_dir, mask_dir, csv_file = fold_dirs[tf]

            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    valid_scores[row["filename"]] = {
                        "anterior": int(row["anterior_score"]),
                        "posterior": int(row["posterior_score"]),
                    }

            imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
            for img_p in imgs:
                basename = os.path.basename(img_p)
                if basename in valid_scores:
                    valid_base_images.append(img_p)
                    valid_mask_paths[img_p] = os.path.join(
                        mask_dir, basename.replace(".jpg", "_mask.png")
                    )

        print(f"Valid source images collected: {len(valid_base_images)} (from Folds {train_folds})")

        # 3. Generate for each class and side
        for target_score in [0, 1, 2, 3, 4]:
            for target_side in [0, 1]:
                side_str = "anterior" if target_side == 0 else "posterior"

                save_dir = os.path.join(
                    output_base_dir, f"db_for_fold{target_test_fold}",
                    f"class_{target_score}", side_str,
                )
                os.makedirs(save_dir, exist_ok=True)

                success_count = 0
                current_pool = valid_base_images.copy()
                random.shuffle(current_pool)
                pool_index = 0

                pbar = tqdm(
                    total=quota_per_side,
                    desc=f"Fold {target_test_fold} | Class {target_score} | {side_str.capitalize()}",
                    unit="img",
                    colour="green",
                )

                while success_count < quota_per_side:
                    if pool_index >= len(current_pool):
                        print(f"\nWARNING: Source pool exhausted! Produced: {success_count}.")
                        break

                    base_img_path = current_pool[pool_index]
                    pool_index += 1

                    mask_path = valid_mask_paths[base_img_path]
                    if not os.path.exists(mask_path):
                        continue

                    img_basename = os.path.basename(base_img_path)
                    real_s_score = valid_scores[img_basename][side_str]

                    # Load and preprocess
                    try:
                        orig_pil = Image.open(base_img_path).convert("RGB")
                        orig_tensor_01 = transform(orig_pil).unsqueeze(0).to(device)
                        orig_tensor_11 = orig_tensor_01 * 2.0 - 1.0

                        mask_pil = Image.open(mask_path).convert("L")
                        single_mask_np = extract_single_side_mask(mask_pil, target_side)
                        mask_tensor = torch.from_numpy(single_mask_np).unsqueeze(0).unsqueeze(0).to(device)
                    except Exception:
                        continue

                    # Run the Best-of-N pipeline
                    try:
                        final_img, final_score, is_success = run_medical_editing_pipeline(
                            flow_model=flow_model,
                            seg_model=seg_model,
                            reward_model=reward_model,
                            orig_tensor_11=orig_tensor_11,
                            mask_tensor=mask_tensor,
                            s_score=real_s_score,
                            t_score=target_score,
                            side=target_side,
                            best_threshold=filter_threshold,
                            num_variants=8,
                            device=device,
                        )

                        if is_success:
                            abs_hash = uuid.uuid4().hex[:6]
                            save_name = (
                                f"Gen_{img_basename.replace('.jpg', '')}"
                                f"_s{real_s_score}_t{target_score}_side{target_side}_{abs_hash}.png"
                            )
                            save_path = os.path.join(save_dir, save_name)
                            vutils.save_image(final_img, save_path, normalize=False)

                            success_count += 1
                            pbar.update(1)
                            pbar.set_postfix({"Latest Score": f"{final_score:.2f}"})

                    except Exception as e:
                        print(f"\nERROR (image: {img_basename}): {e}")
                        torch.cuda.empty_cache()
                        continue

                pbar.close()

    print("\nAll 3-fold offline database construction complete.")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Total: 3 folds × 5 classes × 2 sides × 1000 = 30,000 images
    build_offline_database(quota_per_side=1000)
