"""
Re-extract clean transparent logo from the 8K AI-upscaled image.
This gives a much crisper raster master since Real-ESRGAN anime removed
JPEG artifacts and sharpened edges before extraction.
"""
import cv2
import numpy as np
import torch

SRC = "/home/thc1006/dev/logo/logo_4x_aiUpscale_anime.png"
OUT = "/home/thc1006/dev/logo/logo_clean_8k_aiUpscaled.png"

img = cv2.imread(SRC, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Source: {img.shape}")

# Use the GPU for the per-pixel distance computation - faster on big image
device = torch.device("cuda")
t = torch.from_numpy(img).float().to(device)  # H,W,3

# Sample bg color from corners
corners = torch.cat([
    t[:200, :200].reshape(-1, 3),
    t[:200, -200:].reshape(-1, 3),
    t[-200:, :200].reshape(-1, 3),
    t[-200:, -200:].reshape(-1, 3),
])
bg_color = corners.median(dim=0).values
print(f"BG color: {bg_color.cpu().numpy()}")

fg_color = torch.tensor([255.0, 255.0, 255.0], device=device)

d_bg = torch.norm(t - bg_color, dim=2)
d_fg = torch.norm(t - fg_color, dim=2)

# Use ratio: how foreground-like
max_d = torch.norm(fg_color - bg_color)
alpha = torch.clamp(d_bg / max_d, 0, 1)

# Output: pure white RGB with computed alpha
H, W = img.shape[:2]
rgb = torch.full((H, W, 3), 255, device=device, dtype=torch.uint8)
alpha_u8 = (alpha * 255.0).to(torch.uint8)
rgba = torch.cat([rgb, alpha_u8.unsqueeze(-1)], dim=-1).cpu().numpy()

# Save BGRA
bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
cv2.imwrite(OUT, bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print(f"Saved 8K clean: {OUT}")

# Also save a 1-bit threshold version for high-quality vectorization
gray_alpha = alpha_u8.cpu().numpy()
_, hard = cv2.threshold(gray_alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("/home/thc1006/dev/logo/logo_mask_8k.png", hard)
inv = 255 - hard
cv2.imwrite("/home/thc1006/dev/logo/logo_mask_8k_inv.png", inv)
print("Saved 8K binary masks for vectorization")
