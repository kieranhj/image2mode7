"""
teletext_decode.py — Reverse-decode a 640×480 PNG render of a Teletext page
back to a raw 1000-byte Mode 7 byte stream.

The images on horsenburger.com are 640×480 pixels, which maps as:
  40 cols  ×  16 px/col  = 640   (exact)
  25 rows  ×  19.2 px/row = 480  (fractional → some rows 19px, some 20px)

Each character cell contains a 2×3 sub-pixel grid (2 wide, 3 tall).
Sub-pixel centres are sampled at:
  cx = sp_col * 8 + 4           (sp_col 0..79)
  cy = round((sp_row + 0.5) * 6.4)   (sp_row 0..74)

After sampling and quantising to the 8-colour Teletext palette, the decoder
reconstructs the Mode 7 byte stream using a greedy state-machine pass over each
row, inserting the minimum necessary colour/background control codes.
"""

import numpy as np
from PIL import Image
import pathlib, sys

# ---------------------------------------------------------------------------
# Teletext constants
# ---------------------------------------------------------------------------

# Palette index → RGB
PALETTE_RGB = np.array([
    [  0,   0,   0],   # 0 black
    [255,   0,   0],   # 1 red
    [  0, 255,   0],   # 2 green
    [255, 255,   0],   # 3 yellow
    [  0,   0, 255],   # 4 blue
    [255,   0, 255],   # 5 magenta
    [  0, 255, 255],   # 6 cyan
    [255, 255, 255],   # 7 white
], dtype=np.float32)

# Colour names for debugging
COL_NAME = ['black','red','green','yellow','blue','magenta','cyan','white']

# Mode 7 control codes
M7_ALPHA_BLACK   = 0x80   # (not in normal character set; use 0x20/space in alpha mode)
M7_GFX_RED       = 0x91   # Set fg = red,     graphics mode
M7_GFX_GREEN     = 0x92
M7_GFX_YELLOW    = 0x93
M7_GFX_BLUE      = 0x94
M7_GFX_MAGENTA   = 0x95
M7_GFX_CYAN      = 0x96
M7_GFX_WHITE     = 0x97
M7_BLACK_BG      = 0x9C   # Set bg = black
M7_NEW_BG        = 0x9D   # Set bg = current fg
M7_HOLD_GFX      = 0x9E
M7_RELEASE_GFX   = 0x9F

M7_CONTIG_GFX    = 0x99   # Switch to contiguous graphics (Set-After)
M7_SEP_GFX       = 0x9A   # Switch to separated graphics (Set-After)

M7_GFX_BASE      = 0x90   # M7_GFX_BASE + colour_index = colour code

# Separated graphics: foreground pixels appear as a blend of fg and bg colour
# (SAA5050 insets each sixel by 1px on all sides; net effect ≈ 50% coverage)
SEP_FG_FACTOR = 128   # 128/255 ≈ 50% fg, 50% bg (matches image2teletext.py)

# Graphics characters in Mode 7 (SAA5050) are in the range 0x20–0x7F when in
# graphics mode.  Bit 5 (0x20) is always set (space = all-off).  The 6 sub-pixel
# bits are packed into bits [0,1,2,3,4,6] of the byte (bit 5 is the 0x20 base):
#
#   byte = 0x20 | (pattern & 0x1F) | ((pattern & 0x20) << 1)
#
# Valid graphics bytes: 0x20–0x3F (bit6=0) and 0x60–0x7F (bit6=1).
# Sub-pixel bit order: bit0=top-left, bit1=top-right, bit2=mid-left,
#                      bit3=mid-right, bit4=bot-left, bit5=bot-right.

# Bit masks for the 6 sub-pixels within the graphics byte (matches GFX_PIXEL_BITS
# in image2teletext.py: [1, 2, 4, 8, 16, 64]).
GFX_PIXEL_BITS = [1, 2, 4, 8, 16, 64]


def pattern_to_gfx_byte(pattern6):
    """Convert 6-bit sub-pixel pattern to Mode 7 graphics character byte."""
    low5 = pattern6 & 0x1F
    bit5 = (pattern6 >> 5) & 1
    return 0x20 | low5 | (bit5 << 6)


def gfx_byte_to_pattern(byte):
    """Convert Mode 7 graphics character byte to 6-bit sub-pixel pattern.
    Returns None for non-graphics bytes."""
    if not (0x20 <= byte <= 0x3F or 0x60 <= byte <= 0x7F):
        return None
    low5 = byte & 0x1F
    bit5 = (byte >> 6) & 1
    return low5 | (bit5 << 5)


def is_gfx_byte(byte):
    """Return True if byte is a valid Mode 7 graphics character."""
    return 0x20 <= byte <= 0x3F or 0x60 <= byte <= 0x7F


# ---------------------------------------------------------------------------
# Image loading and sub-pixel sampling
# ---------------------------------------------------------------------------

N_COLS, N_ROWS = 40, 25
SP_COLS, SP_ROWS = N_COLS * 2, N_ROWS * 3   # 80 × 75 sub-pixel grid


def load_and_quantise(img_path):
    """Load a PNG and quantise every pixel to the nearest Teletext palette colour.
    Returns (colour_map H×W uint8, img_w, img_h)."""
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size
    arr = np.array(img, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    dists = np.sum((flat[:, None, :] - PALETTE_RGB[None, :, :]) ** 2, axis=2)
    colour_map = np.argmin(dists, axis=1).reshape(img_h, img_w).astype(np.uint8)
    return colour_map, img_w, img_h


def sample_subpixels(colour_map, img_w, img_h):
    """
    Sample the 80×75 sub-pixel grid from a colour-index map of any size.
    Uses fractional centre sampling — robust to non-integer cell dimensions.
    Returns a (75, 80) array of palette indices.
    """
    px_per_sp_col = img_w / SP_COLS
    px_per_sp_row = img_h / SP_ROWS
    sp = np.zeros((SP_ROWS, SP_COLS), dtype=np.uint8)
    for sr in range(SP_ROWS):
        cy = min(int(round((sr + 0.5) * px_per_sp_row)), img_h - 1)
        for sc in range(SP_COLS):
            cx = min(int(round((sc + 0.5) * px_per_sp_col)), img_w - 1)
            sp[sr, sc] = colour_map[cy, cx]
    return sp


# ---------------------------------------------------------------------------
# Row decoder
# ---------------------------------------------------------------------------

def decode_row(sp_row_block):
    """
    Decode one character row from its 3×80 sub-pixel block.

    sp_row_block: ndarray shape (3, 80), dtype uint8, values 0–7.

    Returns: list of 40 byte values (the Mode 7 character codes for this row).

    Strategy
    --------
    We make a forward pass tracking the streaming decoder state
    (fg_colour, bg_colour).  For each cell we observe the 6 sub-pixel colours
    and determine:

      * The dominant "foreground" colour (the colour that appears in sub-pixels
        that are "lit" relative to the background).
      * The 6-bit lit/unlit pattern.

    When the required fg_colour for a cell differs from the current streaming
    fg_colour, we insert a colour-change control code at the previous cell (if
    it was a blank control-code slot) or at the current cell at the cost of
    losing one cell to the control code.

    NEW_BG is recognised when a cell's background colour appears to change to
    a non-black colour: we emit a NEW_BG code (which sets bg=current_fg) after
    setting fg to the desired new background colour.
    """
    # --- Pass 1: determine the observed fg/bg colour for each cell --------
    cell_fg  = np.zeros(N_COLS, dtype=np.uint8)   # dominant non-bg colour
    cell_bg  = np.zeros(N_COLS, dtype=np.uint8)   # background colour
    cell_pat = np.zeros(N_COLS, dtype=np.uint8)   # 6-bit gfx pattern

    for col in range(N_COLS):
        sp = sp_row_block[:, col * 2:(col + 1) * 2]   # 3×2
        pixels = sp.flatten()                           # 6 values

        vals, counts = np.unique(pixels, return_counts=True)
        if len(vals) == 1:
            # All same colour: treat as background-filled cell (blank / control code)
            cell_bg[col]  = vals[0]
            cell_fg[col]  = vals[0]   # unknown fg, same as bg for now
            cell_pat[col] = 0
        else:
            # Two colours: darker/less-common is bg unless one of them is black
            # Prefer black as bg; otherwise choose the less-frequent colour as bg
            if 0 in vals:
                bg_idx = 0
            else:
                bg_idx = vals[np.argmin(counts)]
            fg_idx = vals[vals != bg_idx][0]
            # Build 6-bit pattern: 1 where pixel == fg_idx
            pat = 0
            for bit, px in enumerate(pixels):
                if px == fg_idx:
                    pat |= (1 << bit)
            cell_fg[col]  = fg_idx
            cell_bg[col]  = bg_idx
            cell_pat[col] = pat

    # --- Pass 2: reconstruct streaming state and emit bytes ---------------
    out = []
    cur_fg = 7   # white
    cur_bg = 0   # black

    col = 0
    while col < N_COLS:
        obs_fg = int(cell_fg[col])
        obs_bg = int(cell_bg[col])
        pat    = int(cell_pat[col])

        # Background mismatch: we need a NEW_BG to change bg.
        # NEW_BG sets bg = cur_fg, so we need cur_fg == obs_bg first.
        if obs_bg != cur_bg and pat == 0 and obs_fg == obs_bg:
            # Blank cell, possibly a control code cell for NEW_BG or colour change.
            # Will be resolved below.
            pass

        if obs_fg != cur_fg and obs_fg != obs_bg:
            # Need a colour change.  Emit it here (costs this cell).
            out.append(M7_GFX_BASE + obs_fg)
            cur_fg = obs_fg
            # This cell is now consumed by the control code; move on.
            col += 1
            continue

        if obs_bg != cur_bg:
            # Background needs to change.
            # To set bg = obs_bg using NEW_BG: we need cur_fg == obs_bg.
            if cur_fg != obs_bg:
                # First emit a colour code to set fg = obs_bg (costs a cell)
                out.append(M7_GFX_BASE + obs_bg)
                cur_fg = obs_bg
                col += 1
                # Now emit NEW_BG (costs another cell)
                if col < N_COLS:
                    out.append(M7_NEW_BG)
                    cur_bg = obs_bg
                    # Restore fg to what we need for the next graphics cell.
                    # (will be handled when we hit the next cell)
                    col += 1
                continue
            else:
                # cur_fg already == obs_bg; emit NEW_BG
                out.append(M7_NEW_BG)
                cur_bg = obs_bg
                col += 1
                continue

        # Normal graphics cell
        out.append(pattern_to_gfx_byte(pat))
        col += 1

    # Pad / truncate to exactly 40 bytes
    while len(out) < N_COLS:
        out.append(0x20)
    return out[:N_COLS]


# ---------------------------------------------------------------------------
# Full image decoder
# ---------------------------------------------------------------------------

def decode_image(img_path):
    """
    Decode a 640×480 Teletext render PNG to a 1000-byte Mode 7 page.
    Returns a bytearray of length 1000.
    """
    colour_map, img_w, img_h = load_and_quantise(img_path)
    sp_grid    = sample_subpixels(colour_map, img_w, img_h)
    result     = bytearray(1000)
    for row in range(N_ROWS):
        block = sp_grid[row * 3:(row + 1) * 3, :]
        row_bytes = decode_row(block)
        result[row * N_COLS:(row + 1) * N_COLS] = row_bytes
    return result


# ---------------------------------------------------------------------------
# Render decoded bytes back to PNG for comparison
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Alpha character glyphs (loaded lazily from teletext2.ttf)
# ---------------------------------------------------------------------------

ALPHA_CELL_W = 12
ALPHA_CELL_H = 20
RENDER_W = N_COLS * ALPHA_CELL_W   # 480
RENDER_H = N_ROWS * ALPHA_CELL_H   # 500

_ALPHA_GLYPHS: dict[int, np.ndarray] | None = None


def _find_font() -> pathlib.Path | None:
    # Source-tree layout (this repo, or editable install).
    p = pathlib.Path(__file__).parent / "datasets" / "teletext_assets" / "teletext2.ttf"
    if p.exists():
        return p
    # Installed via pip — font ships in the image2teletext_assets package.
    try:
        from importlib.resources import files
        p = files("image2teletext_assets").joinpath("teletext2.ttf")
        if p.is_file():
            return pathlib.Path(str(p))
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        pass
    return None


def _load_alpha_glyphs() -> dict[int, np.ndarray]:
    """Rasterise alpha characters (0x20-0x7F) from teletext2.ttf into 12x20
    boolean bitmaps. Cached after first call. Returns {} if the font is missing."""
    global _ALPHA_GLYPHS
    if _ALPHA_GLYPHS is not None:
        return _ALPHA_GLYPHS
    font_path = _find_font()
    if font_path is None:
        _ALPHA_GLYPHS = {}
        return _ALPHA_GLYPHS

    from PIL import ImageFont, ImageDraw
    # Find the size where one character spans ~12x20. The teletext fonts use
    # a fixed character cell, so a single binary search nails it.
    target_w, target_h = ALPHA_CELL_W, ALPHA_CELL_H
    chosen_size = target_h
    for size in range(target_h, target_h + 12):
        try:
            font = ImageFont.truetype(str(font_path), size=size)
            bbox = font.getbbox("M")
            w = bbox[2] - bbox[0]
            if w >= target_w:
                chosen_size = size
                break
        except Exception:
            continue
    font = ImageFont.truetype(str(font_path), size=chosen_size)

    glyphs: dict[int, np.ndarray] = {}
    canvas_w, canvas_h = target_w * 2, target_h * 2
    for byte in range(0x20, 0x7F + 1):
        img = Image.new("L", (canvas_w, canvas_h), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), chr(byte), fill=255, font=font)
        arr = np.array(img) > 127
        # Locate inked pixels and centre them in a target_h x target_w cell
        ys, xs = np.where(arr)
        out = np.zeros((target_h, target_w), dtype=bool)
        if len(xs):
            x_off = max(0, (xs.min() + xs.max() + 1) // 2 - target_w // 2)
            y_off = max(0, (ys.min() + ys.max() + 1) // 2 - target_h // 2)
            cropped = arr[y_off:y_off + target_h, x_off:x_off + target_w]
            ch, cw = cropped.shape
            out[:ch, :cw] = cropped
        glyphs[byte] = out
    _ALPHA_GLYPHS = glyphs
    return _ALPHA_GLYPHS


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _paint_cell_alpha(out, row, col, byte, fg_idx, bg_idx, glyphs):
    y0 = row * ALPHA_CELL_H
    x0 = col * ALPHA_CELL_W
    fg = PALETTE_RGB[fg_idx].astype(np.uint8)
    bg = PALETTE_RGB[bg_idx].astype(np.uint8)
    glyph = glyphs.get(byte)
    if glyph is None:
        out[y0:y0 + ALPHA_CELL_H, x0:x0 + ALPHA_CELL_W] = bg
        return
    out[y0:y0 + ALPHA_CELL_H, x0:x0 + ALPHA_CELL_W] = np.where(
        glyph[..., None], fg, bg
    )


def _paint_cell_gfx(out, row, col, byte, fg_idx, bg_idx, sep):
    y0 = row * ALPHA_CELL_H
    x0 = col * ALPHA_CELL_W
    cx = ALPHA_CELL_W // 2
    cy1 = round(ALPHA_CELL_H * 1 / 3)
    cy2 = round(ALPHA_CELL_H * 2 / 3)
    sp_y = [(0, cy1), (cy1, cy2), (cy2, ALPHA_CELL_H)]
    sp_x = [(0, cx), (cx, ALPHA_CELL_W)]
    for sr, (yy0, yy1) in enumerate(sp_y):
        for sc, (xx0, xx1) in enumerate(sp_x):
            sixel_idx = sr * 2 + sc
            mask = GFX_PIXEL_BITS[sixel_idx]
            is_set = bool(byte & mask)
            if is_set and sep:
                f = SEP_FG_FACTOR
                colour = ((f * PALETTE_RGB[fg_idx]
                           + (255 - f) * PALETTE_RGB[bg_idx]) / 255).astype(np.uint8)
            else:
                colour = PALETTE_RGB[fg_idx if is_set else bg_idx].astype(np.uint8)
            out[y0 + yy0:y0 + yy1, x0 + xx0:x0 + xx1] = colour


def _paint_cell_solid(out, row, col, colour_idx):
    y0 = row * ALPHA_CELL_H
    x0 = col * ALPHA_CELL_W
    out[y0:y0 + ALPHA_CELL_H, x0:x0 + ALPHA_CELL_W] = (
        PALETTE_RGB[colour_idx].astype(np.uint8)
    )


def render_bytes(page_bytes, out_w=None, out_h=None):
    """Render a 1000-byte Mode 7 page to an RGB image.

    Renders internally at the canonical 480x500 (12x20 px per cell), then
    optionally resizes to (out_w, out_h) via PIL nearest-neighbour. If
    out_w/out_h are None, the canonical size is returned.

    Tracks alpha vs graphics mode per row so that text-heavy broadcast pages
    render their alphanumeric content via the teletext2.ttf font glyphs;
    graphics-mode cells still use the 2x3 sub-pixel grid.

    Returns a PIL Image.
    """
    glyphs = _load_alpha_glyphs()
    out = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)

    for row in range(N_ROWS):
        cur_fg = 7
        cur_bg = 0
        cur_sep = False
        gfx_mode = False  # spec: rows start in alpha mode
        hold = False
        last_gfx = 0x20

        for col in range(N_COLS):
            b = page_bytes[row * N_COLS + col]

            # --- Set-At control codes ---
            if b == M7_NEW_BG:
                cur_bg = cur_fg
            elif b == M7_BLACK_BG:
                cur_bg = 0
            elif b == M7_HOLD_GFX:
                hold = True
            elif b == M7_RELEASE_GFX:
                hold = False
                last_gfx = 0x20

            is_alpha_colour = 0x81 <= b <= 0x87
            is_gfx_colour = 0x91 <= b <= 0x97
            is_other_control = b in (M7_NEW_BG, M7_BLACK_BG, M7_HOLD_GFX,
                                     M7_RELEASE_GFX, M7_SEP_GFX, M7_CONTIG_GFX)

            # --- Paint this cell ---
            if is_alpha_colour or is_gfx_colour or is_other_control \
                    or (b < 0x20) or (0x80 <= b <= 0x9F):
                # Control / spacing cell: show held gfx (if hold and gfx mode)
                # or background colour.
                if hold and gfx_mode:
                    _paint_cell_gfx(out, row, col, last_gfx,
                                    cur_fg, cur_bg, cur_sep)
                else:
                    _paint_cell_solid(out, row, col, cur_bg)
            elif gfx_mode and is_gfx_byte(b):
                _paint_cell_gfx(out, row, col, b, cur_fg, cur_bg, cur_sep)
                last_gfx = b
            else:
                # Alpha mode (or any non-gfx printable): use the font glyph
                _paint_cell_alpha(out, row, col, b, cur_fg, cur_bg, glyphs)

            # --- Set-After mode/colour transitions ---
            if is_alpha_colour:
                cur_fg = b - 0x80
                gfx_mode = False
            elif is_gfx_colour:
                cur_fg = b - 0x90
                gfx_mode = True

            if b == M7_SEP_GFX:
                cur_sep = True
            elif b == M7_CONTIG_GFX:
                cur_sep = False

    img = Image.fromarray(out)
    if out_w is not None and out_h is not None \
            and (out_w, out_h) != (RENDER_W, RENDER_H):
        img = img.resize((out_w, out_h), Image.NEAREST)
    return img


# ---------------------------------------------------------------------------
# Test harness: compare decode → re-render against original
# ---------------------------------------------------------------------------

def compare_image(img_path, verbose=False):
    """
    Decode img_path, re-render it, and compare pixel-by-pixel with the original
    (after quantising both to the 8-colour palette).

    Returns a dict with:
        total_cells    — 1000
        matching_cells — cells where every sub-pixel matches
        error_cells    — cells with at least one sub-pixel wrong
        match_pct      — percentage
    """
    img_path = pathlib.Path(img_path)
    colour_map, img_w, img_h = load_and_quantise(img_path)
    sp_orig    = sample_subpixels(colour_map, img_w, img_h)

    page = decode_image(img_path)
    # Re-render and re-sample at same sub-pixel positions as the original
    rendered = np.array(render_bytes(page, img_w, img_h).convert('RGB'), dtype=np.float32)
    flat = rendered.reshape(-1, 3)
    dists = np.sum((flat[:, None, :] - PALETTE_RGB[None, :, :]) ** 2, axis=2)
    colour_map2 = np.argmin(dists, axis=1).reshape(img_h, img_w).astype(np.uint8)
    sp_rend = sample_subpixels(colour_map2, img_w, img_h)

    # Compare cell by cell
    matching = error_cells = 0
    bad = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            orig_sp  = sp_orig[row * 3:(row + 1) * 3, col * 2:(col + 1) * 2]
            rend_sp  = sp_rend[row * 3:(row + 1) * 3, col * 2:(col + 1) * 2]
            if np.array_equal(orig_sp, rend_sp):
                matching += 1
            else:
                error_cells += 1
                bad.append((row, col, orig_sp.flatten(), rend_sp.flatten()))

    if verbose and bad:
        print(f'  First 5 mismatches:')
        for row, col, orig, rend in bad[:5]:
            print(f'    cell ({row},{col}): orig={orig} rend={rend}')

    return {
        'total_cells':    N_ROWS * N_COLS,
        'matching_cells': matching,
        'error_cells':    error_cells,
        'match_pct':      100.0 * matching / (N_ROWS * N_COLS),
        'bad_cells':      bad,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Decode a 640×480 Teletext PNG to Mode 7 bytes')
    parser.add_argument('input', help='Input PNG')
    parser.add_argument('-o', '--output', help='Output .bin file')
    parser.add_argument('--preview', metavar='PNG', help='Save re-rendered preview')
    parser.add_argument('--compare', action='store_true', help='Compare decode→render against original')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.compare:
        result = compare_image(args.input, verbose=args.verbose)
        print(f'{args.input}: {result["matching_cells"]}/{result["total_cells"]} cells match '
              f'({result["match_pct"]:.1f}%)')
    else:
        page = decode_image(args.input)
        if args.output:
            pathlib.Path(args.output).write_bytes(page)
            print(f'Written {args.output}')
        if args.preview:
            render_bytes(page).save(args.preview)
            print(f'Preview saved to {args.preview}')
        if not args.output and not args.preview:
            print(f'Decoded {len(page)} bytes. Use -o or --preview to save output.')
