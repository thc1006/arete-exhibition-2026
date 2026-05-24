"""
Refine the binary mask before vectorization:
- Gentle Gaussian blur in pixel space then re-threshold
- This smooths edge granularity without distorting text letterforms
- Compare with un-refined version

Two paths exist:
  (A) Preserve authentic print-edge texture (text version)
  (B) Refine to a clean, designed-logo look (refined version)
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("/home/thc1006/dev/logo")
mask = cv2.imread(str(ROOT / "esa_mask_8K.png"), cv2.IMREAD_GRAYSCALE)
print(f"Mask: {mask.shape}")

# Path B: Smooth the mask
# 1) Gaussian blur with small sigma
# 2) Re-threshold at 128 (mid-gray)
# 3) This rounds off small protrusions and fills small gaps

# Two refinement levels
for sigma, name in [(2.0, "light"), (4.0, "medium")]:
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    _, refined = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    # Final morphological smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    # Inverted for potrace
    inv = 255 - refined
    cv2.imwrite(str(ROOT / f"esa_mask_8K_refined_{name}.png"), refined)
    # Save as PGM for potrace
    from PIL import Image
    Image.fromarray(inv).save(str(ROOT / f"esa_mask_8K_refined_{name}.pgm"))
    print(f"  Refined {name} (sigma={sigma}): {(refined>0).sum()/refined.size*100:.2f}% fg")

print("Refined masks saved.")
