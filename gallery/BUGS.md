# image2teletext — Known Bugs & Algorithm Gaps

Discovered by running the converter against 1871 standard-aspect images from the
horsenburger.com teletext art gallery and comparing rendered output with the
original pixel-for-pixel.

Test method: `python test_gallery.py` — converts each image with image2teletext,
renders the output back to the original dimensions using `teletext_decode.render_bytes`,
and counts character cells where all 6 sub-pixels match.

**Baseline (pre-fix, 200-image sample, cols 0–39):**
- Mean: 40.5%, Median: 36.0%, ≥80%: 7 images, <50%: 156 images

**After render_bytes fix + direct_sample (200 images, cols 0–39):**
- Mean: 96.6%, Median: 96.6%, ≥80%: 200/200, Perfect (≥99.9%): 13

**Current baseline (200 images, cols 1–39 only — col 0 skipped as control code):**
- Mean: 98.3%, Median: 98.7%, ≥80%: 200/200, Perfect (≥99.9%): 39, Min: 93.3%

---

## Bug 1 — Alphanumeric/text characters not supported (HIGH IMPACT)

**Description:** Many gallery artworks use Mode 7 *alphanumeric* characters (letters,
digits, punctuation) as artistic elements — e.g. a maze-like background built from
repeated letters, or decorative text overlaid on graphics. The converter only ever
emits graphics-mode characters (0xA0–0xFF). It has no mechanism to output text
characters.

**Effect:** Teletext art that relies on text mode is reconstructed as the nearest
graphics approximation, which typically produces large black areas where intricate
blue-on-black letter patterns existed. Match scores of 10–20% are common for these
images.

**Example:** `f2baa0_0019b521970a4b15a8c9b695caf0a0fd~mv2.png` — 10.8% match.
Background is a dense maze of blue letters; converter renders it as black.

**Fix direction:** Detect alphanumeric sub-pixel patterns (3×5 glyph within a 2×3
cell does not fit the 6-bit graphics bitmap model) and fall back to the nearest
SAA5050 alphanumeric character code. Would require glyph lookup against the SAA5050
ROM character table.

---

## Bug 2 — Left-edge control-code cell shows as black, not background colour (MEDIUM IMPACT)

**Description:** In Mode 7 the first colour-code cell in a row consumes a character
position. The cell visually appears as the *current background colour* at the point
it is processed. Many teletext editors and viewers render this cell as the
*eventual* background colour (after any NEW_BG that follows), making it appear as
if the colour extends to the very edge.

The converter always places the initial fg-colour code at column 0 and the image
starts from column 1, producing a 1-cell-wide black (or bg-coloured) strip at the
left edge of solid-background images.

**Effect:** Images with non-black backgrounds that extend to the left margin show a
visible 1-cell-wide black strip in converter output. This accounts for 1–2% match
loss on images with solid non-black backgrounds.

**Example:** `f2baa0_0effd30874424f558cea7c7392ec4587~mv2.png` (91.8% match) — red
background extends to left edge in original; converter shows black col-0 cell.

**Revised understanding:** Per the Teletext specification all control-code state
resets at the start of every row — hold graphics included. There is therefore no
valid mechanism to make col 0 display anything other than the initial background.
The converter output is correct. The gallery artists most likely cropped the
exported PNG to remove the leftmost black strip before uploading, which is why the
original appears to have colour extending to x=0.

**Action:** No fix needed in the converter. `test_gallery.py` now skips col 0
(range starts at 1, denominator is `N_ROWS × (N_COLS - 1) = 975`), consistent
with `make_gallery_html.py`. **Fixed in test_gallery.py (March 2026).**

---

## Bug 3 — Separated graphics not auto-detected (LOW IMPACT after fix)

**Description:** The `--sep` flag enables separated graphics mode globally. Many
gallery artworks mix separated and contiguous graphics within the same image —
using separated blocks for texture/shadow and contiguous for solid fills. The
converter applies one mode uniformly.

**Effect (revised):** With `direct_sample=True` and `snap_palette=True`, all pixel
colours are snapped to pure palette values before the DP runs.  Separated graphics
rendered as a 50% fg/bg blend round to roughly the same palette colour as the pure
fg, so the comparison metric can't distinguish them.  In practice, enabling
`--sep` on the 20 worst-scoring gallery images improves scores by 0.0–1.1%,
not the 5–15% originally estimated.  The effect is visible in the preview but
not in the sub-pixel metric.

**Fix (partial):** `--sep` already allows the DP to choose sep/contiguous
automatically per row section — it emits `SEP_GFX` only where error is reduced.
Additionally, `teletext_decode.render_bytes` now correctly renders separated
graphics (was silently treating them as contiguous). **Fixed in teletext_decode.py
(March 2026).**

**Remaining:** No further algorithmic change needed for the gallery metric.  For
real photographic sources (not palette-quantised), enabling `--sep` may improve
texture rendering; recommend trying it with `--smooth 2`.

---

## Bug 4 — ~~Sub-pixel sampling offset from resize~~ FIXED (LOW-MEDIUM IMPACT)

**Description:** The converter resizes the source image to `pw × ph` pixels using
bilinear (or other) interpolation, then treats each pixel as one sub-pixel. PIL's
bilinear resize samples at `(i + 0.5) / out_size * in_size - 0.5` for output pixel
`i`. For gallery images (~980px wide → 78px), the sample centre for sub-pixel 0 is
`0.5/78 * 980 - 0.5 ≈ 5.8px`. The correct centre for sub-pixel 0 of the 40-column
grid is `(980/80) * 0.5 ≈ 6.1px`. The ~0.3px offset is small but accumulates
across edges, causing occasional wrong sub-pixel samples at colour boundaries.

Additionally, the gallery renders have x=0 of each row permanently black (a
renderer border artefact). PIL bilinear for sub-pixel 0 samples around x≈6, safely
inside the actual content — but `filter='nearest'` would sample at x=0 and always
return black for the first sub-pixel of every row.

**Effect:** 1–3% match loss from boundary misalignment on images with many vertical
colour transitions.

**Fix direction:** Pre-sample the source image using the same centre-sampling
formula as `teletext_decode.sample_subpixels()` before passing to the solver,
rather than relying on PIL resize.

**Fix:** `--direct-sample` flag (also `direct_sample=True` in API) implements
exact centre-point sampling at the 80×75 SP grid positions, bypassing bilinear
resize entirely. Used by default in `test_gallery.py`. **Fixed in image2teletext.py
(March 2026).**

---

## Bug 5 — ~~Hold graphics not emitted~~ NOT A BUG

**Note:** The converter already emits `MODE7_HOLD_GFX` (0x9E) and `MODE7_RELEASE_GFX`
(0x9F). Both are full candidates in the DP solver, the Set-At behaviour is correctly
modelled in the error table, and `--nohold` disables the optimisation. Hold graphics
is therefore fully supported. Removing from the bug list.

---

## Bug 6 — PAR must be derived per-image for gallery renders (OPERATIONAL)

**Description:** The default PAR (1.2) is designed for converting photos to
teletext. Gallery images are already teletext renders and have their own implicit
PAR based on the render dimensions. Using PAR=1.2 on a 978×876 image produces
35 columns instead of 40.

**Fix:** For gallery images, compute `par = (MODE7_PIXEL_H / MODE7_PIXEL_W) *
(img_w / img_h)`. This is implemented in `test_gallery.py:par_for_image()`.
Not a converter bug per se, but a documentation/usability gap.

---

## Bug 7 — ~~teletext_decode.render_bytes used wrong graphics byte range~~ FIXED

**Description:** `teletext_decode.py` originally encoded graphics characters in the
range 0xA0–0xFF (with `pattern_to_gfx_byte` using base `0xA0`), and `render_bytes`
only recognised that range as graphics. However, `image2teletext.py` correctly uses
the SAA5050 encoding where graphics bytes sit in 0x20–0x3F and 0x60–0x7F (base 0x20,
with sub-pixels in bits [0,1,2,3,4,6]). Every graphics cell was therefore rendered
as background, producing completely garbled comparison images and invalid match
scores.

**Effect:** All gallery match scores were systematically wrong — showing 40.5% mean
when the true value (post-fix) is ~56.2% on the same sample.

**Fix:** Changed `pattern_to_gfx_byte` to use base `0x20`; updated `gfx_byte_to_pattern`
to handle 0x20–0x3F and 0x60–0x7F; rewrote `render_bytes` to use `GFX_PIXEL_BITS`
[1,2,4,8,16,64] for bit extraction and added proper hold-graphics state tracking.
**Fixed in teletext_decode.py (March 2026).**

---

## Summary Table

| # | Issue | Impact | Fix Complexity |
|---|-------|--------|----------------|
| 1 | Alphanumeric characters not supported | Very high | High |
| 2 | Left-edge strip — gallery images are cropped; converter is correct | Test artefact | Fixed in test_gallery.py |
| 3 | Separated graphics — render_bytes fix + impact reassessed | Low (was Medium) | Fixed in teletext_decode.py |
| 4 | ~~Sub-pixel sampling offset from resize~~ — FIXED via `--direct-sample` | Was Low-Medium | Fixed |
| 5 | ~~Hold graphics not emitted~~ — NOT A BUG, already implemented | — | — |
| 6 | PAR must be computed per gallery image | Operational | Low (documented) |
| 7 | ~~render_bytes used wrong graphics byte range~~ — FIXED | Was very high | Fixed |
