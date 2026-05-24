"""Compare vtracer and potrace SVG renders at fine-detail crop."""
import cv2, numpy as np

def comp(img, color_bgr):
    if img.shape[2] == 4:
        bgr = img[:, :, :3]; a = img[:, :, 3:4].astype(np.float32) / 255.0
    else:
        bgr = img; a = np.ones((*img.shape[:2], 1), dtype=np.float32)
    bg = np.zeros_like(bgr); bg[..., :] = color_bgr
    return (bgr * a + bg * (1 - a)).astype(np.uint8)

# Source potrace was traced on inverted mask, so SVG is black emblem on white
# vtracer was traced on white emblem on black, so SVG is white emblem
for name in ["vtracer", "potrace"]:
    img = cv2.imread(f"/home/thc1006/dev/logo/_render_{name}.png", cv2.IMREAD_UNCHANGED)
    print(f"{name}: shape={img.shape}, dtype={img.dtype}, channels={img.shape[2] if len(img.shape)==3 else 1}")
    composed = comp(img, (200, 0, 255))
    H = composed.shape[0]
    y0, x0 = int(H * 0.30), int(H * 0.35)
    crop = composed[y0:y0+500, x0:x0+500]
    cv2.imwrite(f"/home/thc1006/dev/logo/_svg_crop_{name}.png", crop)
