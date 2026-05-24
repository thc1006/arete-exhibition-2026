"""
Clean background removal for white-on-navy emblem.

Strategy: per-pixel alpha = color distance to background / (distance to bg + distance to fg).
This preserves anti-aliased edges without thresholding artifacts.
"""
import cv2
import numpy as np
from pathlib import Path

SRC = Path("/home/thc1006/dev/logo/338133880_891341015492742_7414906034460799206_n.jpg")
OUT_DIR = Path("/home/thc1006/dev/logo")

img = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Source: {img.shape} dtype={img.dtype}")

H, W = img.shape[:2]

# Sample background (corners) and foreground (center)
corners = np.concatenate([
    img[0:50, 0:50].reshape(-1, 3),
    img[0:50, -50:].reshape(-1, 3),
    img[-50:, 0:50].reshape(-1, 3),
    img[-50:, -50:].reshape(-1, 3),
])
bg_color = np.median(corners, axis=0).astype(np.float32)
print(f"Detected background color (RGB): {bg_color}")

# Foreground = white (the emblem is solid white)
fg_color = np.array([255.0, 255.0, 255.0], dtype=np.float32)

# Compute per-pixel distances in RGB
pix = img.astype(np.float32)
d_bg = np.linalg.norm(pix - bg_color, axis=2)
d_fg = np.linalg.norm(pix - fg_color, axis=2)

# Alpha = how "foreground-like" each pixel is
# alpha in [0, 1], 0 = pure background, 1 = pure foreground
alpha = d_bg / (d_bg + d_fg + 1e-8)

# Normalize: stretch so that pure bg => 0, pure fg => 1
max_bg_to_fg = np.linalg.norm(fg_color - bg_color)
alpha = np.clip(d_bg / max_bg_to_fg, 0, 1)

# Set RGB to white everywhere (since the logo is monochrome white)
out_rgb = np.full_like(img, 255, dtype=np.uint8)
out_alpha = (alpha * 255.0).astype(np.uint8)

# Compose RGBA
rgba = np.dstack([out_rgb, out_alpha])

# Save high-fidelity transparent PNG at native resolution
out_path = OUT_DIR / "logo_clean_native.png"
# Convert RGBA -> BGRA for cv2
bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
cv2.imwrite(str(out_path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
print(f"Saved: {out_path}")

# Also save a hard-mask binary version for vectorization (no anti-alias)
# Use Otsu threshold on alpha for clean binary mask
_, hard_mask = cv2.threshold(out_alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold cut at: alpha~={(hard_mask>0).sum()/hard_mask.size*100:.2f}% foreground")

# Save hard-edge PNG for tracing (white on transparent)
hard_rgba = np.dstack([np.full((H, W, 3), 255, dtype=np.uint8), hard_mask])
hard_bgra = cv2.cvtColor(hard_rgba, cv2.COLOR_RGBA2BGRA)
hard_path = OUT_DIR / "logo_mask_binary.png"
cv2.imwrite(str(hard_path), hard_bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
print(f"Saved: {hard_path}")

# Also save a black-on-white mask for potrace (potrace expects this)
trace_input = OUT_DIR / "logo_for_potrace.pbm"
# Invert: white emblem -> black on white background
inv = 255 - hard_mask
cv2.imwrite(str(OUT_DIR / "logo_for_potrace.png"), inv)
print(f"Saved tracing input")
