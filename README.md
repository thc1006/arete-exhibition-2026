# arete-exhibition-2026

Reproducible image pipeline that prepares the official logos used in the 2026 **arete** exhibition — turning raster sources (JPEG screenshots, scanned stamps) into clean **transparent PNG** at 2K / 4K / 8K and editable **SVG** vectors.

Two organizations are featured:

| Folder | Organization | Source character |
|--------|--------------|------------------|
| [`01_paichuan_bachelor/`](01_paichuan_bachelor/) | **百川學士學位學程** (NYCU Paichuan Bachelor Degree Program) | High-resolution white emblem on navy — solid two-tone, JPEG ringing on gear teeth |
| [`02_nctu_alumni/`](02_nctu_alumni/) | **交大校友會** (NCTU Alumni Association, "ESA · SINCE 1896") | Photographed/scanned blue stamp on white paper — print noise, contains the year **1896** that must not be distorted |

Built end-to-end around an RTX 5090 (Vulkan), but the vector stage is CPU-only. Each logo finishes in seconds.

## TL;DR — what the pipeline does

| # | Stage | Tool | Why |
|---|-------|------|-----|
| 1 | Source analysis | OpenCV | Auto-detect background color + sharpness; decide whether AI upscale is needed |
| 2 | AI super-resolution (when needed) | Real-ESRGAN `x4plus-anime` via NCNN-Vulkan | Cleans JPEG/scan artifacts and sharpens edges. **Skip when source is already clean digital art** — AI can hallucinate detail |
| 3 | Color-distance alpha matte | NumPy | Per-pixel α = ‖p − bg‖ / ‖fg − bg‖ keeps anti-aliasing; far better than a hard threshold |
| 4 | Refinement (per content) | OpenCV (sigmoid + Gaussian + morphology) | Suppress speckle without distorting letterforms |
| 5 | Vectorization | potrace 1.16 | Smooth curve fitting; tune `--turdsize`, `--alphamax`, `--opttolerance` per logo |
| 6 | Scale-out rendering | Inkscape 1.2 CLI | Lossless raster output at any target size from the SVG |

Per-logo deliverables: 2–4 SVG variants (white / black / brand color) × 2K · 4K · 8K PNG (padded and tightly cropped) + one AI-upscaled raster master as a safety net.

## Repository layout

```
.
├── 01_paichuan_bachelor/        百川學士學位學程
│   ├── source/                  original JPEG
│   ├── svg/
│   │   ├── paichuan_white.svg   ← primary (matches source — white logo)
│   │   └── paichuan_black.svg   ← recolored for editing / dark-on-light placement
│   └── png/
│       ├── 2K.png · 4K.png · 8K.png                  white fill on transparent (square)
│       ├── 2K_cropped.png · 4K_cropped.png · 8K_cropped.png   tight crop
│       └── 4K_black.png · 8K_black.png · ...         black variants
│
├── 02_nctu_alumni/              交大校友會 (ESA SINCE 1896 ALUMNI)
│   ├── source/                  original JPEG
│   ├── svg/
│   │   ├── nctu_alumni_blue.svg      ← primary (refined, brand blue #040B79)
│   │   ├── nctu_alumni_white.svg     ← for dark backgrounds
│   │   ├── nctu_alumni_black.svg     ← for editing / dark-on-light placement
│   │   └── nctu_alumni_authentic.svg ← preserves original print-edge texture
│   └── png/
│       ├── 2K.png · 4K.png · 8K.png                  blue fill (default)
│       ├── 2K_white.png · ... · 8K_white_cropped.png  white variants
│       ├── 2K_black.png · ... · 8K_black.png         black variants
│       └── _alt_8K_AI_raster.png                     AI-upscaled raster master
│
└── _intermediate/               working files, scripts, ESRGAN binary
    ├── scripts/                 the Python scripts that actually ran
    ├── paichuan_files/          intermediate outputs for logo 1
    ├── nctu_alumni_files/       intermediate outputs for logo 2
    ├── renders/                 verification crops and previews
    └── tools/                   Real-ESRGAN NCNN-Vulkan binary
```

**Picking a file** — the safest pairing for most uses:

- Web / print on light background: `*_blue.svg` (NCTU) or `*_white.svg` rendered onto your own background (Paichuan).
- Web / print on dark background: `*_white.svg`.
- Re-coloring or further editing: `*_black.svg`.
- Highest-quality raster: `8K_cropped.png`.

## Pipeline detail

### 1. Decide whether to AI-upscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
```

Rules of thumb actually used:

- **Upscale when** the source is < 1500 px on the long side, **or** sharpness < 1500 (JPEG-compressed / scanned).
- **Skip when** the source is already large and crisp (Laplacian variance > 2000) — AI risks hallucinating where there is no real detail to recover.
- After upscaling, **always read back the text/digits**. For the NCTU stamp, the "1896" was visually compared between AI-4× and bicubic-4× crops before committing to the AI path.

### 2. AI super-resolution

The portable NCNN-Vulkan binary runs directly on the RTX 5090 — no PyTorch, no `basicsr`, no Python 3.13 wheel headaches:

```bash
./realesrgan-ncnn-vulkan \
  -i source.jpg -o source_4x.png \
  -n realesrgan-x4plus-anime -s 4 -f png
```

The **anime** model (not the photo one) is the right choice for logos, line art, and stamped prints: it preserves hard edges and treats flat regions as flat.

### 3. Color-distance alpha matte

A hard threshold throws away anti-aliasing and produces jagged edges. The matte we use:

```python
d_bg  = np.linalg.norm(pix - bg_color, axis=2)
max_d = np.linalg.norm(fg_color - bg_color)
alpha = np.clip(d_bg / max_d, 0, 1)
```

Output RGB choice:

- **Uniform `fg_color`** — recommended for the NCTU stamp; removes paper texture and ink-density drift.
- **Original `img` RGB** — preserves the authentic stamped appearance (`nctu_alumni_authentic.svg` is built from this).

### 4. Refinement (per content type)

| Source type | Refinement |
|-------------|------------|
| Two-tone JPEG (Paichuan) | Sigmoid only, no morphology |
| Scanned stamp (NCTU Alumni) | Sigmoid `k=14, x₀=0.30` → Gaussian σ=4 → `MORPH_CLOSE 3×3` |

The sigmoid `α' = 1 / (1 + exp(−k(α − x₀)))` is strictly better than `np.where(α > t, 1, 0)`: it preserves sub-pixel edge transitions while still pushing low-confidence pixels to 0.

### 5. Vectorization

`potrace` outperforms `vtracer` for binary single-color content — roughly 5× smaller SVG with visibly smoother curves. Convert to PGM first, then trace:

```bash
potrace mask.pgm -s -o out.svg \
  --turdsize 4 --alphamax 1.0 --opttolerance 0.2
```

Per-logo tuning that actually mattered:

| Logo | turdsize | alphamax | opttolerance | Reason |
|------|----------|----------|--------------|--------|
| Paichuan | 4 | 1.0 | 0.2 | Large, clean shapes — defaults are fine |
| NCTU Alumni | 4 | 1.0 | 0.2 | After Gaussian σ=4 refinement, defaults are again the sweet spot |

Recolor in one line — no need to re-trace:

```bash
sed 's/fill="#000000"/fill="#ffffff"/g' out.svg > out_white.svg
sed 's/fill="#000000"/fill="#040B79"/g' out.svg > out_blue.svg
```

### 6. Render at any scale

```bash
inkscape logo.svg --export-type=png \
  --export-filename=logo_8K.png \
  --export-width=8192 \
  --export-background-opacity=0
```

Then crop to content for the tight-fit variant:

```python
ys, xs = np.where(img[:, :, 3] > 4)
pad = max(8, (ys.max() - ys.min()) // 50)
crop = img[ys.min() - pad : ys.max() + pad,
           xs.min() - pad : xs.max() + pad]
```

## Per-logo notes

### 01 — 百川學士學位學程 (Paichuan Bachelor Program)
- **Source:** 2005 × 2005, white emblem on navy. Detected `bg = RGB(25, 46, 91)`.
- AI 4× was used because the source JPEG had visible ringing along the gear teeth and the wreath leaves.
- Final SVG ≈ 82 KB. The "white on transparent" is the primary because the original design is white — no recoloring needed for fidelity.

### 02 — 交大校友會 (NCTU Alumni Association)
- **Source:** 2048 × 1699, scanned blue stamp print on white paper.
- Ink `RGB(4, 11, 121)`; background `RGB(254, 254, 254)`.
- **Text/year preservation** drove every decision. "1896", "ESA", "SINCE", "ALUMNI" were all visually verified after AI upscaling.
- Four SVG variants are shipped because they serve different needs:
  - `_blue.svg` — refined edges, brand blue; the canonical asset.
  - `_white.svg` — for dark backgrounds.
  - `_black.svg` — editable starting point.
  - `_authentic.svg` — preserves the print-edge texture for "old stamp" presentations.

## Tools and versions

- **Real-ESRGAN NCNN-Vulkan** `v0.2.5.0` — portable binary, no Python dep
- **potrace** `1.16`
- **Inkscape** `1.2.2`
- **Python 3.13** + `opencv-python`, `numpy`, `Pillow`
- **PyTorch** `2.9.1+cu128` (only used when GPU tensor ops were convenient; the heavy GPU work runs through NCNN-Vulkan)

## Reproducing the pipeline

```bash
# 1. CLI tooling
sudo apt-get install -y potrace inkscape
pip install opencv-python numpy Pillow

# 2. Portable Real-ESRGAN — no Python required
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d realesrgan
chmod +x realesrgan/realesrgan-ncnn-vulkan

# 3. The actual scripts that ran are in _intermediate/scripts/
ls _intermediate/scripts/
```

## Decisions worth flagging

1. **NCNN-Vulkan over PyTorch Real-ESRGAN.** Avoids the `basicsr` / Python 3.13 wheel mess and removes any CUDA-driver coupling. Vulkan works on the 5090 immediately.
2. **potrace over vtracer for binary content.** Roughly 5× smaller SVG and visibly smoother curves. vtracer wins for full-color rasters, not for this.
3. **Sigmoid α, not threshold.** The difference is visible at 8K — especially around small text and gear teeth.
4. **Multiple SVG color variants per logo.** Recoloring SVG is a one-line `sed`, so shipping all useful fills is essentially free and saves the consumer from re-doing the work.
5. **Padded square + tightly cropped PNG pairs.** Designers want both: a square for placement grids, a tight crop for embedding inline.

## License & attribution

This repository ships the **processing pipeline and intermediate artifacts** used to prepare logos for the 2026 arete exhibition.

The two logos themselves remain the intellectual property of their respective owners — 百川學士學位學程 (NYCU Paichuan Bachelor Degree Program) and 交大校友會 (NCTU Alumni Association). They are reproduced here strictly for the exhibition's authorized use; please contact the respective organization before any other use.
