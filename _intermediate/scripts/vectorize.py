"""
Vectorize the high-quality binary mask to SVG using vtracer.
Tuned for clean emblem with mix of curves (wreath leaves, gear circle)
and sharp corners (building columns, sword).
"""
import vtracer
from pathlib import Path
import time

# Trace the 8K binary mask (white emblem on transparent / white on black)
src_8k = "/home/thc1006/dev/logo/logo_mask_8k.png"
src_native = "/home/thc1006/dev/logo/logo_mask_binary.png"

# Run vtracer with tuned params
def trace(input_path, output_path, **kwargs):
    t0 = time.time()
    vtracer.convert_image_to_svg_py(
        input_path,
        output_path,
        colormode=kwargs.get("colormode", "binary"),
        hierarchical=kwargs.get("hierarchical", "stacked"),
        mode=kwargs.get("mode", "spline"),
        filter_speckle=kwargs.get("filter_speckle", 4),
        color_precision=kwargs.get("color_precision", 6),
        layer_difference=kwargs.get("layer_difference", 16),
        corner_threshold=kwargs.get("corner_threshold", 60),
        length_threshold=kwargs.get("length_threshold", 4.0),
        max_iterations=kwargs.get("max_iterations", 10),
        splice_threshold=kwargs.get("splice_threshold", 45),
        path_precision=kwargs.get("path_precision", 3),
    )
    dt = time.time() - t0
    size = Path(output_path).stat().st_size / 1024
    print(f"  {Path(output_path).name}: {size:.1f} KB in {dt:.1f}s")

print("Tracing 8K AI-upscaled mask with vtracer (spline, binary):")
trace(src_8k, "/home/thc1006/dev/logo/logo_8k_spline.svg",
      mode="spline", filter_speckle=8, corner_threshold=70,
      length_threshold=4.0, splice_threshold=45, path_precision=4)

print("Tracing 8K - finer details, lower corner threshold:")
trace(src_8k, "/home/thc1006/dev/logo/logo_8k_spline_detail.svg",
      mode="spline", filter_speckle=4, corner_threshold=50,
      length_threshold=2.0, splice_threshold=30, path_precision=5)

print("Tracing native 2K for comparison:")
trace(src_native, "/home/thc1006/dev/logo/logo_2k_spline.svg",
      mode="spline", filter_speckle=4, corner_threshold=60,
      length_threshold=3.0, splice_threshold=40, path_precision=4)

print("Done.")
