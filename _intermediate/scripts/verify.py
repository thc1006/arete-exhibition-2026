"""Render the transparent PNG over a checker + colored bg to verify edge quality."""
import cv2
import numpy as np
from pathlib import Path

p = Path("/home/thc1006/dev/logo/logo_clean_native.png")
img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
print(f"Loaded: {img.shape}, channels={img.shape[2]}")

H, W = img.shape[:2]
bgr = img[:, :, :3]
alpha = img[:, :, 3].astype(np.float32) / 255.0

# Checker background
sq = 40
yy, xx = np.indices((H, W))
checker = (((yy // sq) + (xx // sq)) % 2).astype(np.uint8)
checker_bg = np.where(checker[..., None] == 0, 200, 240).astype(np.uint8)
checker_bg = np.broadcast_to(checker_bg, (H, W, 3)).copy()

# Composite over checker
comp_checker = (bgr * alpha[..., None] + checker_bg * (1 - alpha[..., None])).astype(np.uint8)
cv2.imwrite("/home/thc1006/dev/logo/_verify_checker.png", comp_checker)

# Composite over magenta (extreme contrast)
mag = np.full_like(bgr, 0, dtype=np.uint8)
mag[..., 2] = 255  # red
mag[..., 0] = 200  # blue (magenta in BGR)
comp_mag = (bgr * alpha[..., None] + mag * (1 - alpha[..., None])).astype(np.uint8)
cv2.imwrite("/home/thc1006/dev/logo/_verify_magenta.png", comp_mag)

# Composite back over original navy color for sanity check
navy = np.full_like(bgr, 0, dtype=np.uint8)
navy[..., 0] = 91  # B
navy[..., 1] = 46  # G
navy[..., 2] = 25  # R
comp_navy = (bgr * alpha[..., None] + navy * (1 - alpha[..., None])).astype(np.uint8)
cv2.imwrite("/home/thc1006/dev/logo/_verify_navy.png", comp_navy)

print("Verification images saved.")
