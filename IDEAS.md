# image2teletext — Ideas & Future Work

Ideas for making the converter produce output that feels more like hand-crafted
teletext art, informed by studying the galleries at horsenburger.com and
teletextart.co.uk.  Roughly prioritised by impact vs. effort.

---

## High impact

### ~~1. Row colour-change minimisation (post-processing pass)~~ ✓ DONE
~~The DP can flip colour every cell; human artists use 0–2 colour transitions per
row, placed at object edges.  A post-processing pass over the finished byte array
that merges short same-colour runs (e.g. two cyan cells surrounded by yellow →
three yellow) would eliminate the "salt-and-pepper" noise that is the most
obvious tell of automated conversion.  No changes to the core solver needed.~~

### 2. Palette discipline — small palettes first
`--quant N` already reduces colours but is treated as optional.  For a teletext-
art look, aggressively reducing to 4–6 dominant colours *before* cell
quantisation — and snapping each region to its nearest Teletext colour — produces
the "deliberate colour composition" look of hand-made art.  Consider making a
low quant value part of the default `art` preset rather than an opt-in extra.

### 3. Background flattening
Human artists almost always collapse the background to a single solid colour.
Detect the background region (largest connected area, or the dominant colour in
the image border) and reduce it to one Teletext colour before conversion.  Even a
crude implementation would produce a large stylistic improvement.

### 4. Silhouette-priority edge weighting
The DP minimises error uniformly across the image.  A saliency or edge-detection
pass could feed per-cell weights into the error metric so that subject/background
boundary cells matter more than internal texture cells.  This preserves clean
outlines even when internal detail is sacrificed.

---

## Medium impact

### 5. Per-region separated vs. contiguous graphics mode
The `--sep` flag applies separated mode globally.  The real artistic technique is
to use contiguous graphics for flat foreground objects and separated graphics for
textured backgrounds, hair, foliage, shadow fringing — anything that shouldn't
be a solid block.  A local-variance pass per character cell could switch mode
automatically: low variance → contiguous, high variance → separated.

### 6. Held graphics at colour transitions
When a colour-change control code is needed mid-row it consumes a character cell,
producing a gap.  Held graphics mode causes that cell to display the last-seen
graphics character instead, creating a smooth-seeming transition.  Implementing
this in the DP/greedy solver would match the technique that skilled artists use
to avoid visual gaps at mid-row colour changes.

### 7. Subject-area detection for portrait/landscape modes
Different source images suit different teletext conventions:
- Portrait → silhouette-with-colour-masses: face region gets one or two colours,
  hair another, background flattened.
- Landscape → 3-band horizontal: sky colour, mid-ground colour, foreground colour.
- Icon/logo → direct sixel mapping with aggressive palette reduction.
Offer a `--style portrait|landscape|icon|auto` flag that applies the appropriate
simplification strategy before the main solver runs.

### 8. Sixel pattern as halftone / brightness suggestion
Within a given foreground colour, denser sixel patterns (more of the 6 bits set)
appear brighter; sparser patterns appear darker.  A converter pass that selects
sixel fill density to suggest luminance within a colour region — rather than
always choosing the nearest-error pattern — would produce the shading vocabulary
that artists use to suggest form and depth.

---

## Lower impact / cosmetic

### 9. Page framing template
Wrap output in the standard teletext page structure: a header row with page
number and service name in alphanumeric text mode, a footer row with fastext-
style coloured link boxes.  Pure cosmetic, no algorithmic complexity, but
immediately makes any output read as a teletext page rather than a raw graphic.

### 10. Double-height support
The SAA5050 supports a double-height mode (a control code that makes a row of
characters span two physical rows).  Using this for large foreground elements
would produce blockier, more monumental compositions — a distinctive teletext
aesthetic.  Currently unsupported.

### 11. Flash/blink support
Characters can be set to flash at ~1 Hz, cycling between visible and invisible.
Used artistically for emphasis and simple animation.  Could be exported as an
animated GIF preview.  Currently unsupported.

### 12. Animated GIF input → animated teletext output
Modern teletext art is frequently published as animated GIFs.  Extend the
converter to accept an animated GIF and produce a multi-frame sequence (one
1000-byte page per frame) that can be replayed as a loop.  The existing
gif2frames.bat suggests this was already being explored.

---

## Notes on the teletext art aesthetic (from gallery research)

- **Dark backgrounds dominate.**  Almost all gallery works use black backgrounds;
  bright colours on black read clearly and feel authentic.
- **Cyan, magenta, yellow are the workhorses.**  White for highlights; red for
  warmth/emphasis; blue for sky/shadow (visually dark, use carefully).
- **Large colour regions, not per-cell colour.**  The streaming constraint is a
  style, not just a limitation.  Embrace horizontal colour banding.
- **Separated graphics for texture.**  The single biggest distinction between
  skilled and naive teletext art.  Solid blocks everywhere is the naive default.
- **Silhouette over internal detail.**  The outline must read clearly; internal
  texture is secondary.  Sacrifice detail to preserve the edge.
- **Economy of means.**  The best teletext art asks: what are the 3–5 most
  important visual elements, and how can each be expressed with a handful of
  cells?  More faithful ≠ better looking.
