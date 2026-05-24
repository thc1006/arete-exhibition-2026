"""Quality comparison: crop the gear teeth area from AI vs bicubic at 8K."""
import cv2
import numpy as np

ai = cv2.imread("/home/thc1006/dev/logo/logo_clean_8k_aiUpscaled.png", cv2.IMREAD_UNCHANGED)
bi = cv2.imread("/home/thc1006/dev/logo/logo_clean_4x_gpu_bicubic.png", cv2.IMREAD_UNCHANGED)
print(f"AI: {ai.shape}, Bicubic: {bi.shape}")

# Composite on magenta for visibility
def comp(img, color_bgr):
    bgr = img[:, :, :3]
    a = img[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full_like(bgr, 0)
    bg[..., 0] = color_bgr[0]; bg[..., 1] = color_bgr[1]; bg[..., 2] = color_bgr[2]
    return (bgr * a + bg * (1 - a)).astype(np.uint8)

ai_comp = comp(ai, (200, 0, 255))    # magenta in BGR
bi_comp = comp(bi, (200, 0, 255))

# Crop the gear teeth region (top of inner circle)
# At 8020x8020, top of gear is around y=1000-1700, x=3000-5000
# At 8020x8020 for AI, and 4010x4010 for bicubic (which is 4x of 1002 not 2005)
# Wait - bicubic 4x of 2005 is 8020.  Both should be 8020.

# Get top center crop where gear teeth are
def crop(img, name):
    H = img.shape[0]
    y0 = int(H * 0.30)
    x0 = int(H * 0.35)
    y1 = y0 + 800
    x1 = x0 + 800
    crop = img[y0:y1, x0:x1]
    cv2.imwrite(f"/home/thc1006/dev/logo/_crop_{name}.png", crop)
    print(f"_crop_{name}.png saved {crop.shape}")

crop(ai_comp, "ai")
crop(bi_comp, "bicubic")
