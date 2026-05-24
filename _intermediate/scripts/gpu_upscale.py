"""
GPU-accelerated high-quality upscale of the cleaned logo.
Uses PyTorch on the RTX 5090 for Lanczos/bicubic resampling.

For a clean two-color graphic, AI upscaling (Real-ESRGAN) is sub-optimal
because it hallucinates texture; high-order interpolation preserves
the geometric shapes faithfully. The "infinite resolution" path is SVG.
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time

device = torch.device("cuda")
print(f"Device: {device} - {torch.cuda.get_device_name(0)}")
print(f"VRAM free: {torch.cuda.mem_get_info(0)[0] / 1e9:.2f} GB")

src = "/home/thc1006/dev/logo/logo_clean_native.png"
img = cv2.imread(src, cv2.IMREAD_UNCHANGED)  # BGRA
print(f"Source: {img.shape}")

# Move to GPU as float tensor [1, C, H, W]
t = torch.from_numpy(img).float().to(device)
t = t.permute(2, 0, 1).unsqueeze(0)  # 1,4,H,W

H, W = t.shape[-2:]

def gpu_resize(tensor, scale, mode):
    new_H, new_W = int(tensor.shape[-2] * scale), int(tensor.shape[-1] * scale)
    return F.interpolate(tensor, size=(new_H, new_W), mode=mode, align_corners=False if mode != "nearest" else None)

# Run several scales for "highest quality" PNG output set
scales = {
    "2x": 2,
    "4x": 4,
}

for name, s in scales.items():
    torch.cuda.synchronize(); t0 = time.time()
    up = gpu_resize(t, s, "bicubic")
    up = torch.clamp(up, 0, 255)
    torch.cuda.synchronize(); dt = time.time() - t0
    arr = up.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    out = f"/home/thc1006/dev/logo/logo_clean_{name}_gpu_bicubic.png"
    cv2.imwrite(out, arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f"  {name:>3} bicubic GPU resize {H}x{W} -> {arr.shape[0]}x{arr.shape[1]}  in {dt*1000:.1f} ms  -> {out}")

# Also produce a smooth-mask version: run on alpha channel only with higher-quality
# resample (area for downscale or precise bicubic), then re-combine
print("\nFinal native-resolution clean PNG already at:", src)
print("GPU upscaling complete.")
