"""Verify the ESA extraction quality by compositing and cropping text regions."""
import cv2
import numpy as np

img = cv2.imread("/home/thc1006/dev/logo/esa_clean_8K.png", cv2.IMREAD_UNCHANGED)
H, W = img.shape[:2]
print(f"8K clean: {img.shape}")

def comp(img, color_bgr):
    bgr = img[:, :, :3]
    a = img[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.zeros_like(bgr); bg[..., :] = color_bgr
    return (bgr * a + bg * (1 - a)).astype(np.uint8)

# Composite on YELLOW (contrasting with blue) for max edge visibility
yellow_comp = comp(img, (0, 255, 255))
cv2.imwrite("_verify_esa_yellow.png", yellow_comp)

# Crop the "1896" region from the verification image
y0, y1 = 3700, 4250
x0, x1 = 3300, 5400
cv2.imwrite("_verify_1896.png", yellow_comp[y0:y1, x0:x1])

# Crop "ALUMNI"
cv2.imwrite("_verify_alumni.png", yellow_comp[5100:6100, 2400:6300])

# Whole preview at lower res
preview = cv2.resize(yellow_comp, (1200, int(1200 * H / W)))
cv2.imwrite("_verify_preview.png", preview)
print("Verification images saved")
