"""Test if AI upscaling preserves the '1896' digits correctly vs bicubic."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Bicubic 4x via GPU for fair comparison
src = cv2.imread("302503360_456138886535623_8655709148618267214_n.jpg")
print(f"Source: {src.shape}")

t = torch.from_numpy(src).float().cuda().permute(2, 0, 1).unsqueeze(0)
up = F.interpolate(t, scale_factor=4, mode="bicubic", align_corners=False)
up = torch.clamp(up, 0, 255)
bicubic = up.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
cv2.imwrite("esa_4x_bicubic.png", bicubic, [cv2.IMWRITE_PNG_COMPRESSION, 1])
print(f"Bicubic 4x: {bicubic.shape}")

# Compare key text regions: "1896", "ESA", "SINCE", "ALUMNI"
ai = cv2.imread("esa_4x_ai.png")
print(f"AI: {ai.shape}")

# Locate '1896' (rough region — manually estimated then expanded)
# Original image is 2048x1699. "1896" appears around y=950-1050, x=850-1300 in original
# Scale 4x -> y=3800-4200, x=3400-5200
def crop(img, y0, y1, x0, x1, name):
    c = img[y0:y1, x0:x1]
    cv2.imwrite(f"_text_{name}.png", c)
    return c

# At 4x
crop(ai,      3700, 4250, 3300, 5400, "1896_ai")
crop(bicubic, 3700, 4250, 3300, 5400, "1896_bicubic")

# "ALUMNI" - bottom: original y=1300-1500, x=600-1500 -> 4x: y=5200-6000, x=2400-6000
crop(ai,      5100, 6100, 2400, 6300, "alumni_ai")
crop(bicubic, 5100, 6100, 2400, 6300, "alumni_bicubic")

# "ESA" upper center: y=350-700, x=750-1300 -> 4x: y=1400-2800, x=3000-5200
crop(ai,      1300, 2900, 2800, 5400, "esa_letters_ai")
crop(bicubic, 1300, 2900, 2800, 5400, "esa_letters_bicubic")

print("Text crops saved for visual inspection.")
