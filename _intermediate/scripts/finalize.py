"""
Produce final deliverables:
- Tightly cropped versions removing transparent padding
- Validation composite showing both white and black variants
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("/home/thc1006/dev/logo")

def crop_to_alpha(img):
    """Crop image to its non-transparent bounding box, with small padding."""
    if img.shape[2] != 4:
        return img
    alpha = img[:, :, 3]
    ys, xs = np.where(alpha > 4)
    if len(ys) == 0:
        return img
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = max(8, (y1 - y0) // 50)
    y0 = max(0, y0 - pad); y1 = min(img.shape[0], y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(img.shape[1], x1 + pad)
    return img[y0:y1, x0:x1]

# Crop the 4K and 8K renders
for sz in ["2K", "4K", "8K"]:
    src = ROOT / f"logo_{sz}.png"
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    cropped = crop_to_alpha(img)
    out = ROOT / f"logo_{sz}_cropped.png"
    cv2.imwrite(str(out), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"  {sz}: {img.shape} -> cropped {cropped.shape}  ({out.stat().st_size/1024:.0f} KB)")

# Make a verification composite: 4K white-fill emblem on navy
img = cv2.imread(str(ROOT / "logo_4K_cropped.png"), cv2.IMREAD_UNCHANGED)
H, W = img.shape[:2]
navy = np.zeros((H, W, 3), dtype=np.uint8)
navy[..., 0] = 91; navy[..., 1] = 46; navy[..., 2] = 25  # BGR navy
bgr = img[:, :, :3]
a = img[:, :, 3:4].astype(np.float32) / 255.0
final = (bgr * a + navy * (1 - a)).astype(np.uint8)
cv2.imwrite(str(ROOT / "_final_verify_4K_on_navy.png"), final, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print("Final verification on navy saved.")

# Also produce a black-fill variant (often more useful for designers)
# Re-render the black SVG at 4K too
import subprocess
subprocess.run(["inkscape", str(ROOT / "logo.svg"),
                "--export-type=png", f"--export-filename={ROOT}/logo_4K_black.png",
                "--export-width=4096", "--export-background-opacity=0"],
               check=False, capture_output=True)
print("Black-fill 4K rendered.")
