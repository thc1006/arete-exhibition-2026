"""
Extract Excel Biomedical logo from white background.

This source is already clean (digital, not scanned), so:
- Minimal refinement (don't over-smooth fine text strokes)
- Use grayscale luminance directly as alpha (simpler, faithful)
- Preserve all dots and thin strokes
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path("/home/thc1006/dev/logo")
SRC = ROOT / "excel_4x_ai.png"

img = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
H, W = img.shape[:2]
print(f"Source: {img.shape}")

# Convert to grayscale (luminance)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Alpha = 255 - gray  (since black=ink, white=bg)
# This naturally preserves anti-aliased edges
alpha = 255 - gray

# Optional: sigmoid contrast to push partial-alpha noise to 0
# Less aggressive than for the ESA logo because this source is much cleaner
af = alpha.astype(np.float32) / 255.0
k = 10.0
x0 = 0.20
af = 1.0 / (1.0 + np.exp(-k * (af - x0)))
alpha_clean = (af * 255).astype(np.uint8)

# Output: pure black RGB with alpha
black_rgb = np.zeros((H, W, 3), dtype=np.uint8)
rgba = np.dstack([black_rgb, alpha_clean])
bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

out = ROOT / "excel_clean_8K.png"
cv2.imwrite(str(out), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print(f"Saved clean: {out}  ({out.stat().st_size/1e6:.2f} MB)")

# Hard binary mask for vectorization
_, hard = cv2.threshold(alpha_clean, 128, 255, cv2.THRESH_BINARY)
# Light morphology to close pixel gaps (small kernel since source is already clean)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel)

fg_pct = (hard > 0).sum() / hard.size * 100
print(f"Binary mask: {fg_pct:.2f}% foreground")
cv2.imwrite(str(ROOT / "excel_mask_8K.png"), hard)

# Inverted PGM for potrace (potrace expects dark=trace, light=bg)
Image.fromarray(255 - hard).save(str(ROOT / "excel_mask_8K.pgm"))
print("Saved binary mask + PGM")

# Verification composite
def comp(img, color):
    bgr = img[:,:,:3]; a = img[:,:,3:4].astype(np.float32)/255.0
    bg = np.zeros_like(bgr); bg[...,:] = color
    return (bgr*a + bg*(1-a)).astype(np.uint8)

# Yellow background for max contrast with black logo
verify = comp(bgra, (0, 255, 255))
preview = cv2.resize(verify, (1200, int(1200 * H / W)))
cv2.imwrite("_excel_verify_preview.png", preview)
# Crop icon area to verify dots
cv2.imwrite("_excel_verify_icon.png", verify[1100:1850, 50:1200])
cv2.imwrite("_excel_verify_chinese.png", verify[600:1120, 200:2200])
print("Verifications saved.")
