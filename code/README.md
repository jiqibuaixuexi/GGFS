# GGFS — Training & Inference

This directory contains the core training script and end-to-end inference pipeline for the **GGFS** (Guided Generation and Filtering System) framework. For the full project overview, architecture, and clinical results, see the [main README](../README.md).


## Dependencies

### External Modules (required by `infer.py`)

| Module | Provides |
|--------|----------|
| `models.inpainting_flow_model_DAG` | `InpaintingFlowUNet` |
| `models.SegModel` | `ResUNet34` (anatomy-aware critic) |
| `models.RewardModel` | `PathologyRewardModel` (ResNet-34 evaluator) |
| `utils.sampling` | `sample_energy_guided_cfg_editing` |
| `utils.tools` | `extract_single_side_mask` |

### Packages

```
torch torchvision numpy pillow tqdm pandas
```

## Usage

### Training

```bash
python train.py
```

The training script performs CFG pretraining with independent spatial (mask) and semantic (pathology score) dropout, producing the velocity-field predictor used by the CNF ODE solver.

### Inference — Single Image (Best-of-N)

```python
from infer import generate_n_variants, run_medical_editing_pipeline

variants = generate_n_variants(flow_model, seg_model, img, mask, src, tgt, side)
best_img, score, ok = run_medical_editing_pipeline(
    flow_model, seg_model, reward_model, img, mask, src, tgt, side, threshold
)
```

### Inference — Offline Database Construction (3-Fold Cross-Validation)

```bash
python infer.py
```

Produces `offline_database/db_for_fold{1,2,3}/class_{0–4}/{anterior,posterior}/` with UUID-named synthetic CT images for HEM augmentation.

## Checkpoint Layout

```
checkpoints/
├── inpainting_flow_mse/
│   ├── train12_val3/inpainting_flow_epoch_50.pt
│   ├── train13_val2/inpainting_flow_epoch_50.pt
│   └── train23_val1/inpainting_flow_epoch_50.pt
├── seg_model_checkpoints/
│   ├── t12_v3/best_resunet_34.pth
│   ├── t13_v2/best_resunet_34.pth
│   └── t23_v1/best_resunet_34.pth
└── reward_model_checkpoints/
    ├── filter_fold1_best.pth
    ├── filter_fold2_best.pth
    └── filter_fold3_best.pth
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drop_mask_prob` | 0.1 | Spatial CFG dropout rate (training) |
| `drop_score_prob` | 0.1 | Semantic CFG dropout rate (training) |
| `energy_scale` | 50.0 | Morphological energy guidance step size |
| `guidance_start` | 0.1 | ODE timestep to begin energy guidance |
| `guidance_end` | 0.4 | ODE timestep to end energy guidance |
| `gap_scale` | 0.1 | Gap-region fraction for bone bridge dismantling |
| `num_variants` | 8 | Candidate variants per image (production) |
| `quota_per_side` | 1000 | Target images per (fold, class, anatomical side) |
