# image2teletext — Known Bugs & Algorithm Gaps

Discovered by running the converter against 1871 standard-aspect images from the
horsenburger.com teletext art gallery and comparing rendered output with the
original pixel-for-pixel.

Test method: `python test_gallery.py` — converts each image with image2teletext,
renders the output back to the original dimensions using `teletext_decode.render_bytes`,
and counts character cells where all 6 sub-pixels match.

**Baseline results (200-image sample, May 2026):**
- Mean cell match: 40.5%
- Median: 36.0%
- ≥80% match: 7 images (3.5%)
- <50% match: 156 images (78%)

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

**Fix direction:** Implement hold-graphics support (`MODE7_HOLD_GFX`). With hold
active, the colour-code cell at col 0 displays the last-held graphics character
instead of background. If the previous row's last character was a solid-bg char,
and bg=red, the held character would also appear red — matching the original.

---

## Bug 3 — Separated graphics not auto-detected (MEDIUM IMPACT)

**Description:** The `--sep` flag enables separated graphics mode globally. Many
gallery artworks mix separated and contiguous graphics within the same image —
using separated blocks for texture/shadow and contiguous for solid fills. The
converter applies one mode uniformly.

**Effect:** Images whose artist used separated graphics for backgrounds, hair,
foliage, or shadow fringing are reproduced with solid contiguous blocks instead,
losing the characteristic texture. Typically 5–15% match loss.

**Fix direction:** Per-cell local-variance detection (IDEAS.md #5): cells with high
colour variance → separated mode; low variance → contiguous. Requires the solver to
try both modes per cell and pick the lower-error option.

---

## Bug 4 — Sub-pixel sampling offset from resize (LOW-MEDIUM IMPACT)

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

---

## Bug 5 — Hold graphics control-code cell reconstruction (LOW IMPACT)

**Description:** When `MODE7_HOLD_GFX` (0x9E) is active, control-code cells
display the last graphics character rather than the background. Some artists use
this to make mid-row colour transitions appear seamless (the colour-code cell
visually continues the previous graphics character). The converter never emits
`MODE7_HOLD_GFX` or `MODE7_RELEASE_GFX`, so it cannot reproduce this effect.

**Effect:** Mid-row colour transitions have a 1-cell-wide background-coloured gap
in converter output where the original is seamless. Mainly affects images with
dense multi-colour rows.

**Fix direction:** IDEAS.md #6 — held graphics at colour transitions. Add HOLD_GFX
to the DP state and allow it to be emitted when the hold-character matches the
error budget.

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

## Summary Table

| # | Issue | Impact | Fix Complexity |
|---|-------|--------|----------------|
| 1 | Alphanumeric characters not supported | Very high | High |
| 2 | Left-edge colour-code cell shows as black | Medium | Medium (hold gfx) |
| 3 | Separated graphics not auto-detected | Medium | Medium |
| 4 | Sub-pixel sampling offset from resize | Low-Medium | Low |
| 5 | Hold graphics not emitted | Low | Medium |
| 6 | PAR must be computed per gallery image | Operational | Low (documented) |
