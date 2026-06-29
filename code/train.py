"""
GGFS — Training Script (Continuous Normalizing Flow)
======================================================
Core model definition and CNF training loop for controllable syndesmophyte editing
in AS (Ankylosing Spondylitis) lumbar CT images.

Model: 7-channel Flow Matching UNet with DAG-based pathology embedding,
anatomical side conditioning, and decoupled Classifier-Free Guidance (CFG)
training with independent spatial and semantic dropout.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ==============================================================================
# Embeddings
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DAGOrdinalWithNullEmbedding(nn.Module):
    """
    DAG-based progressive feature encoding for syndesmophyte severity.
    Encodes scores 0–4 via learned basis vectors and a composition mask.
    Index 5 is reserved as the CFG null-condition embedding.
    """
    def __init__(self, embed_dim=128, null_index=5):
        super().__init__()
        self.null_index = null_index

        # 4 basis features: [Base, Top Spur, Bottom Spur, Bridge]
        self.basis = nn.Parameter(torch.randn(4, embed_dim))

        # Dedicated null embedding for CFG dropout
        self.null_embed = nn.Parameter(torch.randn(1, embed_dim))

        # Composition mask: rows = class 0–4, columns = [base, top, bottom, bridge]
        composition_mask = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Class 0 (00): base structure only
            [1.0, 1.0, 0.0, 0.0],  # Class 1 (10): base + top spur
            [1.0, 0.0, 1.0, 0.0],  # Class 2 (01): base + bottom spur
            [1.0, 1.0, 1.0, 0.0],  # Class 3 (11): base + top + bottom spurs
            [1.0, 1.0, 1.0, 1.0],  # Class 4 (22): base + top + bottom + bridge
        ])
        self.register_buffer('composition_mask', composition_mask)

    def forward(self, scores):
        # scores: [B]
        safe_scores = torch.where(scores == self.null_index, torch.zeros_like(scores), scores)
        safe_scores = safe_scores.long()

        # Look up feature activation mask: [B, 4]
        batch_masks = self.composition_mask[safe_scores]

        # Weighted sum of basis vectors: [B, embed_dim]
        ordinal_out = torch.matmul(batch_masks, self.basis)

        # Replace null-condition samples with the learned null embedding
        is_null = (scores == self.null_index).float().unsqueeze(-1)  # [B, 1]
        out = ordinal_out * (1.0 - is_null) + self.null_embed * is_null

        return out


# ==============================================================================
# UNet Building Blocks
# ==============================================================================

class ResidualBlock(nn.Module):
    """2D residual block with time embedding conditioning."""
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.time_emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.residual_conv = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)
        h = self.conv1(x)
        t_emb = self.time_emb_proj(t_emb)
        h = h + t_emb[:, :, None, None]
        h = self.conv2(h)
        return h + residual


class SelfAttention2D(nn.Module):
    """2D self-attention block operating on spatial feature maps."""
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.softmax(
            torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1
        )
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return x + out


class DownBlock(nn.Module):
    """UNet down-sampling block with optional attention."""
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim) for i in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, t_emb):
        skips = []
        for block in self.blocks:
            x = block(x, t_emb)
            skips.append(x)
        x = self.attn(x)
        x_down = self.downsample(x)
        return x_down, skips


class UpBlock(nn.Module):
    """UNet up-sampling block with skip connections."""
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, upsample=True, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim) for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skips, t_emb):
        x = self.upsample(x)
        for block in self.blocks:
            if skips:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
        x = self.attn(x)
        return x


class MidBlock(nn.Module):
    """UNet bottleneck with self-attention."""
    def __init__(self, channels, time_emb_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim) for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(channels)

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.attn(x)
        return x


# ==============================================================================
# Main Model: Inpainting Flow UNet
# ==============================================================================

class InpaintingFlowUNet(nn.Module):
    """
    Core CNF velocity-field predictor for GGFS.
    Accepts 7-channel input (x_t, mask, masked_image) and fuses physical time,
    DAG-based pathology severity, and anatomical side information to predict
    the ODE velocity field dx/dt.
    """
    def __init__(self, img_ch=3, masked_img_ch=3, mask_ch=1, base_ch=128, time_emb_dim=512, num_res_blocks=2):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Pathology condition embedding (0-4 = actual scores, 5 = CFG null)
        self.pathology_emb = DAGOrdinalWithNullEmbedding(embed_dim=time_emb_dim, null_index=5)

        # Anatomical side embedding (0 = anterior, 1 = posterior)
        self.side_emb = nn.Embedding(2, time_emb_dim)

        # Input: x_t (3) + mask (1) + masked_img (3) = 7 channels
        in_channels = img_ch + mask_ch + masked_img_ch
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # UNet backbone
        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, downsample=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, num_res_blocks)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, num_res_blocks)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, num_res_blocks, use_attention=True)

        self.mid = MidBlock(base_ch * 8, time_emb_dim, num_res_blocks * 2)

        self.up4 = UpBlock(base_ch * 8, base_ch * 4, time_emb_dim, num_res_blocks, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, time_emb_dim, num_res_blocks)
        self.up2 = UpBlock(base_ch * 2, base_ch, time_emb_dim, num_res_blocks)
        self.up1 = UpBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, upsample=False)

        # Output: predicted velocity field (dx/dt) -> 3 channels
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_ch, kernel_size=3, padding=1),
        )

    def forward(self, x_t, t, mask, masked_image, pathology_class, side_class):
        """
        Args:
            x_t:             [B, 3, H, W]  Noised image at timestep t
            t:               [B]           Timestep (0..1)
            mask:            [B, 1, H, W]  1 = region to inpaint, 0 = background
            masked_image:    [B, 3, H, W]  Original image with mask region zeroed out
            pathology_class: [B]           syndesmophyte score (0–4)
            side_class:      [B]           0 = anterior, 1 = posterior
        Returns:
            Predicted velocity field [B, 3, H, W]
        """
        # Fuse all conditioning signals
        t_emb = self.time_mlp(t)
        p_emb = self.pathology_emb(pathology_class)
        s_emb = self.side_emb(side_class)
        emb = t_emb + p_emb + s_emb

        # Concatenate spatial inputs
        x = torch.cat([x_t, mask, masked_image], dim=1)
        x = self.init_conv(x)

        # UNet forward
        skips = []
        x, s1 = self.down1(x, emb); skips.extend(s1)
        x, s2 = self.down2(x, emb); skips.extend(s2)
        x, s3 = self.down3(x, emb); skips.extend(s3)
        x, s4 = self.down4(x, emb); skips.extend(s4)

        x = self.mid(x, emb)

        x = self.up4(x, skips, emb)
        x = self.up3(x, skips, emb)
        x = self.up2(x, skips, emb)
        x = self.up1(x, skips, emb)

        return self.final(x)


# ==============================================================================
# Training Loop
# ==============================================================================

def train_inpainting_cfg_flow(
    model,
    train_loader,
    num_epochs=100,
    lr=1e-4,
    drop_mask_prob=0.1,
    drop_score_prob=0.1,
    device="cuda",
    save_dir="./inpainting_flow_checkpoints",
):
    """
    Train the CNF velocity-field predictor with decoupled CFG dropout.

    Uses Flow Matching: sample x_t = t*z + (1-t)*x_0 with target velocity u = z - x_0.
    CFG training drops spatial condition (mask) and semantic condition (pathology score)
    independently, enabling flexible guidance scales at inference.

    Args:
        model: InpaintingFlowUNet instance
        train_loader: DataLoader yielding dict with keys:
                      "image" [B,3,H,W], "mask" [B,1,H,W],
                      "pathology_class" [B], "side_class" [B]
        num_epochs: Number of training epochs
        lr: Learning rate for AdamW
        drop_mask_prob: Probability of zeroing out the mask (spatial CFG dropout)
        drop_score_prob: Probability of zeroing out the pathology score (semantic CFG dropout)
        device: Training device
        save_dir: Directory for saving checkpoints

    Returns:
        Trained model
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    print(f"GGFS CNF Training | Mask Drop={drop_mask_prob}, Score Drop={drop_score_prob}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for batch in loop:
            # Normalize images to [-1, 1]
            images = batch["image"].to(device)
            z = images * 2.0 - 1.0
            masks = batch["mask"].to(device)

            pathology_classes = batch["pathology_class"].to(device)
            side_classes = batch["side_class"].to(device)
            B = z.shape[0]

            # Build masked context: zero out inpaint region
            masked_images = z * (1.0 - masks)

            # CNF Flow Matching trajectory: x_t = t*z + (1-t)*x_0, target velocity = z - x_0
            x_0 = torch.randn_like(z)
            t = torch.rand(B, device=device, dtype=torch.float32)
            t_broadcast = t.view(B, 1, 1, 1)
            x_t = t_broadcast * z + (1.0 - t_broadcast) * x_0
            u_target = (z - x_0).detach()

            # ------- CFG: Independent spatial & semantic dropout -------
            drop_mask_flag = torch.rand(B, device=device) < drop_mask_prob
            drop_score_flag = torch.rand(B, device=device) < drop_score_prob

            # Spatial dropout: zero both mask and masked_image
            drop_mask_flag_view = drop_mask_flag.view(B, 1, 1, 1)
            train_masks = torch.where(drop_mask_flag_view, torch.zeros_like(masks), masks)
            train_masked_images = torch.where(
                drop_mask_flag_view, torch.zeros_like(masked_images), masked_images
            )

            # Semantic dropout: set score to null index 5
            null_scores = torch.full_like(pathology_classes, 5)
            train_scores = torch.where(drop_score_flag, null_scores, pathology_classes)

            # Forward + loss
            optimizer.zero_grad()
            pred_u = model(
                x_t=x_t, t=t,
                mask=train_masks, masked_image=train_masked_images,
                pathology_class=train_scores, side_class=side_classes,
            )
            loss = mse(pred_u, u_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / len(train_loader):.4f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"inpainting_flow_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)

    return model


# ==============================================================================
# Main — Example Training Entry Point
# ==============================================================================

if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from dataset.dataset import AS_Inpainting_Dataset

    # --- Paths (update as needed) ---
    CSV_FILE   = r"data/cv_ds/t12_v3/train_labels_inpainting.csv"
    IMG_DIR    = r"data/cv_ds/t12_v3/train_bbox"
    MASK_DIR   = r"data/cv_ds/t12_v3/train_point_based_masks"
    BATCH_SIZE = 8
    NUM_EPOCHS = 50

    img_transform = transforms.Compose([transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = AS_Inpainting_Dataset(
        csv_file=CSV_FILE, img_dir=IMG_DIR, point_mask_dir=MASK_DIR,
        transform=img_transform, mask_transform=mask_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = InpaintingFlowUNet(img_ch=3, masked_img_ch=3, mask_ch=1, base_ch=128).to(device)

    trained_model = train_inpainting_cfg_flow(
        model=model,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        drop_mask_prob=0.1,
        drop_score_prob=0.1,
        device=device,
        save_dir="checkpoints/inpainting_flow_mse",
    )
