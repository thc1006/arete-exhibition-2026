"""
Clean extraction of the ESA emblem (blue ink on white paper).

Strategy:
- Sample white background and blue ink color
- Per-pixel alpha = closeness to ink color (not white)
- Output RGB = uniform clean blue everywhere (since emblem is monochrome ink)
  This removes paper texture noise while preserving exact edge shapes.
- Hard-mask version for SVG tracing.
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("/home/thc1006/dev/logo")
SRC = ROOT / "esa_4x_ai.png"

img = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
print(f"Source: {img.shape}")

# Sample background (corners, large patches)
bg_samples = np.concatenate([
    img[:300, :300].reshape(-1, 3),
    img[:300, -300:].reshape(-1, 3),
    img[-300:, :300].reshape(-1, 3),
    img[-300:, -300:].reshape(-1, 3),
])
bg_color = np.median(bg_samples, axis=0).astype(np.float32)
print(f"Background (white paper) RGB: {bg_color}")

# Sample foreground ink from a wide area, taking the darkest pixels
# Look at the gear/anvil area which is densely inked
center_patch = img[H//3:H*2//3, W//3:W*2//3].reshape(-1, 3)
brightness = center_patch.sum(axis=1)
darkest_idx = np.argsort(brightness)[:5000]   # top 5000 darkest in center
fg_samples = center_patch[darkest_idx]
fg_color = np.median(fg_samples, axis=0).astype(np.float32)
print(f"Foreground (ink) RGB: {fg_color}")

# Per-pixel: alpha based on how "ink-like" each pixel is
# Use distance to background normalized by bg-to-fg distance
pix = img.astype(np.float32)
d_bg = np.linalg.norm(pix - bg_color, axis=2)
max_d = np.linalg.norm(fg_color - bg_color)
alpha = np.clip(d_bg / max_d, 0, 1)

# Smooth slight edges - small gaussian helps with noise on the alpha
alpha_u8 = (alpha * 255).astype(np.uint8)

# Compose with UNIFORM ink color
clean_rgb = np.broadcast_to(fg_color.astype(np.uint8).reshape(1, 1, 3), (H, W, 3)).copy()

rgba = np.dstack([clean_rgb, alpha_u8])
bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
out_path = ROOT / "esa_clean_8K.png"
cv2.imwrite(str(out_path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print(f"Saved 8K clean: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

# Also keep ORIGINAL colors version (preserves authentic ink variations)
rgba_orig = np.dstack([img, alpha_u8])
bgra_orig = cv2.cvtColor(rgba_orig, cv2.COLOR_RGBA2BGRA)
out2 = ROOT / "esa_clean_8K_origColor.png"
cv2.imwrite(str(out2), bgra_orig, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print(f"Saved 8K original-color: {out2}  ({out2.stat().st_size/1e6:.1f} MB)")

# Binary masks for vectorization
# Use Otsu to pick threshold automatically
_, hard = cv2.threshold(alpha_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
fg_pct = (hard > 0).sum() / hard.size * 100
print(f"Binary mask: {fg_pct:.2f}% foreground")

cv2.imwrite(str(ROOT / "esa_mask_8K.png"), hard)
# Inverted (black emblem on white) for potrace
cv2.imwrite(str(ROOT / "esa_mask_8K_inv.png"), 255 - hard)
print("Binary masks saved.")
