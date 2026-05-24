"""
Improved extraction with cleaner alpha:
- Use sigmoid contrast curve to push partial-alpha noise to 0
- Apply small median filter to remove speckle noise
- Preserve sharp edges with bilateral filter on alpha
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("/home/thc1006/dev/logo")
SRC = ROOT / "esa_4x_ai.png"

img = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

# Sample bg and fg colors
bg_color = np.array([254., 254., 254.], dtype=np.float32)
fg_color = np.array([4., 11., 121.], dtype=np.float32)
print(f"BG: {bg_color}, FG: {fg_color}")

# Per-pixel "ink-ness"
pix = img_rgb.astype(np.float32)
d_bg = np.linalg.norm(pix - bg_color, axis=2)
max_d = np.linalg.norm(fg_color - bg_color)
alpha_raw = np.clip(d_bg / max_d, 0, 1)

# Apply sigmoid contrast - sharpens the transition near 0.5
# alpha' = 1 / (1 + exp(-k*(alpha - x0)))
# k controls steepness, x0 the midpoint
k = 14.0
x0 = 0.30   # Push midpoint down so partial-alpha noise -> 0
alpha = 1.0 / (1.0 + np.exp(-k * (alpha_raw - x0)))
alpha = (alpha * 255).astype(np.uint8)

# Gentle median filter on alpha - removes single-pixel noise without blurring edges
alpha_clean = cv2.medianBlur(alpha, 3)

# Build the cleaned RGBA with uniform ink color
clean_rgb = np.broadcast_to(fg_color.astype(np.uint8).reshape(1, 1, 3), (H, W, 3)).copy()
rgba = np.dstack([clean_rgb, alpha_clean])
bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

out = ROOT / "esa_clean_8K.png"
cv2.imwrite(str(out), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print(f"Saved cleaned 8K: {out}  ({out.stat().st_size/1e6:.1f} MB)")

# Binary mask for vectorization
# Use threshold high enough to exclude noise (e.g., 128)
_, hard = cv2.threshold(alpha_clean, 128, 255, cv2.THRESH_BINARY)

# Morphological cleanup: close tiny gaps in dense ink, open tiny speckles
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel)
hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, kernel)

fg_pct = (hard > 0).sum() / hard.size * 100
print(f"Binary mask: {fg_pct:.2f}% foreground")

cv2.imwrite(str(ROOT / "esa_mask_8K.png"), hard)
cv2.imwrite(str(ROOT / "esa_mask_8K_inv.png"), 255 - hard)

# Verification composite on yellow
def comp(img, color_bgr):
    bgr = img[:, :, :3]
    a = img[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.zeros_like(bgr); bg[..., :] = color_bgr
    return (bgr * a + bg * (1 - a)).astype(np.uint8)

verify = comp(bgra, (0, 255, 255))
cv2.imwrite("_verify_v2_full.png", verify)
cv2.imwrite("_verify_v2_1896.png", verify[3700:4250, 3300:5400])
cv2.imwrite("_verify_v2_alumni.png", verify[5100:6100, 2400:6300])
cv2.imwrite("_verify_v2_esa.png", verify[1300:2900, 2800:5400])
preview = cv2.resize(verify, (1200, int(1200 * H / W)))
cv2.imwrite("_verify_v2_preview.png", preview)
print("Verify saved.")
