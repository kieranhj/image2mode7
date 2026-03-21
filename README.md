# image2mode7

© 2026 Kieran Connell. Co-written with [Claude](https://claude.ai) (Anthropic).
Based on an original algorithm by Julian Brown (aka Puppeh).

Converts any image to BBC Micro **Mode 7 (Teletext)** format — a 40×25 character
page encoded as a 1000-byte binary file compatible with the BBC Micro, emulators,
and online Teletext editors.

Mode 7 uses the SAA5050 Teletext chip: each character cell is 2×3 sub-pixels,
drawn from a palette of 8 colours (black, red, green, yellow, blue, magenta, cyan,
white).  Colour changes are encoded as in-band control codes that consume a
character cell and affect all subsequent cells on the same row — so the converter
must solve a constrained optimisation problem, not just remap pixels.

## Components

| File | Purpose |
|------|---------|
| `image2teletext.py` | Command-line converter (Python) |
| `image2teletext_ui.py` | Gradio web UI |
| `image2mode7.exe` | Legacy C++ converter (basic, superseded by the Python tool) |

## Quick start

```
pip install pillow numpy
python image2teletext.py myimage.png --preview out.png
```

Output is `myimage.png.bin` — a raw 1000-byte Mode 7 page.

## Command-line reference

```
python image2teletext.py [options] input
```

### Output

| Option | Description |
|--------|-------------|
| `-o OUTPUT` | Output `.bin` file (default: `<input>.bin`) |
| `--preview PNG` | Save a rendered preview as a PNG |
| `--url` | Print an [edit.tf](https://edit.tf) URL for the output |
| `--zxnet` | Print a ZXNet teletext editor URL for the output |
| `--ssd DISK.SSD` | Add output to a BBC Micro DFS `.ssd` disk image (created if absent) |
| `--ssd-name NAME` | BBC DFS filename on the disk (max 7 chars) |

### Solver

| Option | Description |
|--------|-------------|
| `--greedy` | Fast greedy solver instead of full DP. Applies local-search refinement automatically; ~4× faster than DP, good for quick previews |
| `--refine` | Run a local-search refinement pass after solving. Tries every valid candidate at each cell position and accepts substitutions that reduce tail error. 1–3 passes to convergence |
| `--nohold` | Disable Hold Graphics optimisation |
| `--nofill` | Disable New Background optimisation |
| `--sep` | Enable Separated Graphics mode |

### Error metric

| Option | Description |
|--------|-------------|
| `--luma` | Use perceptual luminance weighting (ITU-R BT.601) |
| `--linear` | Linearise sRGB pixels before computing squared error (corrects gamma bias in dark tones) |

### Image pre-processing

| Option | Description |
|--------|-------------|
| `--filter {bilinear,lanczos,bicubic,nearest,cimg}` | Resampling filter for resize (default: `bilinear`) |
| `--par RATIO` | Pixel aspect ratio — `1.2` modern LCD TV (default), `1.0` square pixels / emulator, `1.22` CRT TV |
| `--contrast C` | Contrast factor before resize (1.0 = off; 1.2–1.5 subtle; 1.5–2.5 strong) |
| `--gamma G` | Power-law tone adjustment: `output = (input/255)^(1/G) × 255`. 1.5–2.2 brightens; 0.5–0.8 darkens |
| `--saturation S` | Colour saturation factor (1.0 = off; 1.5–2.0 recommended for photos) |
| `--sharpen-radius R` | Unsharp mask blur radius in pixels (default: 1.0) |
| `--sharpen-amount PCT` | Unsharp mask strength as a percentage (0 = off; 100–200 recommended) |
| `--sharpen-threshold T` | Minimum per-channel difference before sharpening is applied (0 = everywhere; 3–10 skips flat areas) |
| `--median RADIUS` | Median filter of `(2R+1)²` pixels before resize; removes noise/JPEG artefacts (0 = off; 1–3 typical) |
| `--posterize BITS` | Posterise to N bits per channel (0 = off; 3–4 typical) |

### Palette & colour

| Option | Description |
|--------|-------------|
| `--quant N` | Pre-quantise to N colours after resize (0 = off; 8–32 for photos). Uses greedy furthest-first diversity selection to avoid dominant-colour bias |
| `--snap T` | Snap pixels within Euclidean RGB distance T to the nearest Teletext colour (0 = off; 20–40 subtle; 60–80 aggressive) |
| `--snap-palette` | After `--quant`, snap every colour region unconditionally to its nearest Teletext palette colour — the "art palette" look. Pairs well with `--quant 4–8` |
| `--bg-flatten T` | Flatten background before resize. Samples image border to detect dominant background colour, then replaces all border-connected pixels within distance T with the nearest Teletext colour (0 = off; 40 conservative; 60 moderate; 80+ aggressive) |
| `--dither` | Floyd-Steinberg error diffusion at sub-pixel level after resize |
| `--smooth N` | Merge colour runs shorter than N cells into the dominant neighbouring colour after solving (0 = off; 2–3 subtle; 4–6 bold/hand-drawn look) |

### Presets

Named bundles of options for common use cases.  Any explicit flag overrides the preset.

```
--preset {art,clean,crt,dark,flat,graphic,photo,retro,smooth,tv,vivid}
```

| Preset | Best for |
|--------|---------|
| `photo` | Balanced portraits and landscapes |
| `clean` | Like `photo` with light denoising and snap (less speckle) |
| `smooth` | Heavy noise reduction, noisy JPEGs |
| `vivid` | Punchy colours, strong edges |
| `graphic` | Logos and cartoons (tight sharpening, high saturation) |
| `flat` | Bold posterised look with limited palette |
| `retro` | Authentic Ceefax style, blocky limited colours |
| `art` | Teletext art aesthetic: bg flatten, 6-colour palette, snap, high saturation |
| `dark` | Underexposed images (gamma lift) |
| `tv` | LCD TV viewing (PAR 1.2 + photo) |
| `crt` | CRT TV viewing (PAR 1.22 + photo) |

## Web UI

```
pip install gradio pillow numpy
python image2teletext_ui.py
```

Opens a local Gradio interface with sliders for all major options, live preview,
and download buttons for the `.bin` file and rendered PNG.

## Output format

The `.bin` file is a raw 1000-byte sequence: 25 rows × 40 bytes, each byte a
Mode 7 character code.  It can be loaded directly into a BBC Micro (e.g. via
`*LOAD filename 7C00`) or opened in [edit.tf](https://edit.tf) and similar tools.
