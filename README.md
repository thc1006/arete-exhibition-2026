# arete-exhibition-2026

A reproducible pipeline for converting raster logos (JPEG) into clean transparent **PNG** (2K / 4K / 8K) and editable **SVG** vectors. Built around an RTX 5090 (Vulkan) and CPU-only vector tooling, so it runs end-to-end in seconds per logo.

## TL;DR

| Step | Tool | Why |
|------|------|-----|
| 1. Source analysis | OpenCV | Auto-detect background color + sharpness; decide whether AI upscale is needed |
| 2. AI super-resolution (optional) | Real-ESRGAN `x4plus-anime` via NCNN-Vulkan | Cleans JPEG artifacts, sharpens edges. **Skip for clean digital sources** — it can hallucinate detail |
| 3. Color-distance alpha matte | NumPy | Per-pixel `α = ‖p − bg‖ / ‖fg − bg‖` keeps anti-aliasing; far better than hard threshold |
| 4. Refinement (per content) | OpenCV (sigmoid + Gaussian + morphology) | Suppress speckle noise without distorting letterforms |
| 5. Vectorization | potrace 1.16 | Smooth curve fitting; tune `--turdsize`, `--alphamax`, `--opttolerance` per logo |
| 6. Render at scale | Inkscape 1.2 CLI | Lossless raster output at any resolution from SVG |

Deliverables per logo: 2 SVG variants (light/dark fill) × 6 PNG raster files (2K/4K/8K, padded + tightly cropped) × 2–3 color variants. Plus one AI-upscaled raster master as a fallback.

## Final structure

```
.
├── 01_emblem_wreath/         white emblem on navy (source had clean color blocks)
├── 02_esa_alumni/            scanned blue stamp print (noisy, contains text "1896")
├── 03_excel_biomedical/      digital biotech logo (Chinese + English text, dots)
└── _intermediate/            scripts, masks, ESRGAN binary, working files
```

Each logo folder is structured identically:

```
0X_<name>/
├── source/  <original.jpg>
├── svg/     <name>_white.svg, <name>_black.svg, [<name>_brand.svg]
└── png/     2K.png · 4K.png · 8K.png · *_cropped.png · *_white.png · *_black.png
```

## Pipeline

### 1. Decide whether to AI-upscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
H, W = gray.shape
```

Rules of thumb actually used:

- **AI upscale when**: source < 1500 px on the long side, OR sharpness < 1500 (JPEG-compressed / scanned).
- **Skip AI when**: source is already large + clean digital with hard edges (Laplacian variance > 2000).
- Always verify text/digits after AI upscale by cropping and reading them back. Hallucination is the only real risk.

### 2. AI super-resolution (when needed)

Real-ESRGAN's NCNN-Vulkan binary runs on the RTX 5090 without PyTorch, so there is **no CUDA/driver coupling**:

```bash
./realesrgan-ncnn-vulkan \
  -i source.jpg -o source_4x.png \
  -n realesrgan-x4plus-anime -s 4 -f png
```

`x4plus-anime` (not the photo model) is the right pick for logos, line art, and stamped prints — it preserves hard edges and treats flat regions as flat. The PyTorch path is avoided because `basicsr`/`spandrel` have Python 3.13 wheel issues at the time of writing.

### 3. Color-distance alpha matte

A hard threshold throws away anti-aliasing and produces jagged edges. Use a soft matte instead:

```python
d_bg = np.linalg.norm(pix - bg_color, axis=2)
alpha = np.clip(d_bg / np.linalg.norm(fg_color - bg_color), 0, 1)
```

For two-color sources this gives near-perfect edges. Choose the output RGB as either:

- **Uniform `fg_color`** (recommended for monochrome ink/stamp logos): removes paper texture and color drift.
- **Original `img` RGB**: preserves authentic appearance, including ink variation.

### 4. Refinement (per content type)

| Source type | Refinement |
|-------------|------------|
| Clean digital (Excel) | Sigmoid `k=10, x₀=0.20` only |
| Scanned print (ESA) | Sigmoid `k=14, x₀=0.30` + Gaussian σ=4 + `MORPH_CLOSE 3×3` |
| Solid color blocks (emblem) | Sigmoid is enough; no morphology |

A sigmoid (`α' = 1 / (1 + exp(-k(α − x₀)))`) is strictly better than `np.where(α > t, 1, 0)` because it preserves sub-pixel edges while still suppressing low-confidence noise.

### 5. Vectorization

`potrace` outperforms `vtracer` for binary single-color content — it produces ~5× smaller files with smoother curves. Convert to PGM first, then trace:

```bash
potrace mask.pgm -s -o out.svg \
  --turdsize 2 --alphamax 0.8 --opttolerance 0.2
```

Tuning matters per content:

- `--turdsize` = minimum feature area (pixels). Lower it to **2** when small features like dots or thin strokes matter. Default `2` is correct for the Excel logo; raise to `4`–`8` for noisy scans.
- `--alphamax` controls corner detection. `0.8` keeps geometric/sans-serif text sharp; `1.0` allows more rounding for organic shapes.
- `--opttolerance` controls path simplification. `0.1`–`0.2` is the sweet spot.

Recolor by simple `sed` on the resulting SVG:

```bash
sed 's/fill="#000000"/fill="#ffffff"/g' out.svg > out_white.svg
```

### 6. Render at any scale

```bash
inkscape logo.svg --export-type=png \
  --export-filename=logo_8K.png \
  --export-width=8192 \
  --export-background-opacity=0
```

Then crop to content for tighter files:

```python
ys, xs = np.where(img[:,:,3] > 4)
pad = max(8, (ys.max() - ys.min()) // 50)
crop = img[ys.min()-pad:ys.max()+pad, xs.min()-pad:xs.max()+pad]
```

## Per-logo notes

### 01 — emblem wreath
- Source: 2005×2005, white emblem on navy. Detected `bg = RGB(25, 46, 91)`.
- AI 4× chosen because the source JPEG had visible ringing on the gear teeth.
- Vectorized at 8K → potrace 82 KB SVG.

### 02 — ESA Alumni (the hard one)
- Source: 2048×1699, scanned blue stamp print on white.
- Ink color `RGB(4, 11, 121)`; bg `RGB(254, 254, 254)`.
- **Text preservation** was the constraint — "1896" verified by visual comparison of AI vs bicubic 4× crops before committing to AI. The AI version did not distort digits.
- Three SVG variants saved because the user can pick the trade-off:
  - `_blue.svg` — refined, smoothed edges (clean modern look)
  - `_authentic.svg` — preserves original print-edge texture (faithful to the stamp)
  - `_black.svg` / `_white.svg` — single-color reposes

### 03 — Excel Biomedical
- Source: only 591×591, 27 KB heavily compressed. Bimodal histogram (clean two-color).
- AI 4× → 2364×2364. Verified that all four Chinese characters (`昶安生技`) and **three dots** in the icon survived.
- `--turdsize 2` was chosen specifically so the dots are not eaten.
- SVG ~14 KB — by far the smallest, because the source was already two-tone.

## Tools and versions used

- Real-ESRGAN NCNN-Vulkan `v0.2.5.0` (portable binary, no Python dep)
- potrace `1.16`
- Inkscape `1.2.2`
- Python 3.13 + `opencv-python`, `numpy`, `Pillow`, `vtracer` (compared, rejected for binary content)
- PyTorch `2.9.1+cu128` (only used when the GPU is needed for tensor ops; otherwise NCNN-Vulkan does the GPU work)

## Reproducing on a different machine

```bash
# 1. Install CLI tooling
sudo apt-get install -y potrace inkscape
pip install opencv-python numpy Pillow

# 2. Download portable Real-ESRGAN (no Python required)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d realesrgan
chmod +x realesrgan/realesrgan-ncnn-vulkan

# 3. Run a pipeline (see _intermediate/scripts/ for the actual scripts used)
```

## Decisions worth flagging

1. **NCNN-Vulkan over PyTorch Real-ESRGAN** — avoids the `basicsr`/Python-3.13 wheel mess and removes the CUDA/driver coupling. Vulkan works on the 5090 immediately.
2. **potrace over vtracer** for binary content — 5× smaller SVG, visibly smoother curves. vtracer wins for multi-color rasters, not for this.
3. **Sigmoid alpha, not threshold** — the difference is visible at 8K, especially on small text and gear teeth.
4. **Multiple SVG color variants per logo** — recoloring SVG is a one-line `sed`, so shipping both fills is essentially free and saves the consumer from doing it themselves.
5. **Cropped + padded PNG pairs** — designers want both: a square for placement grids, a tight crop for embedding.
