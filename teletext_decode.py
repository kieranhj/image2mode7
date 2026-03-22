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

M7_GFX_BASE      = 0x90   # M7_GFX_BASE + colour_index = colour code

# Graphics character: 0xA0–0xBF (contiguous), 0xE0–0xFF (contiguous)
#   bit pattern: bits 0–5 map to the 6 sub-pixels
#   (bit layout: [top-left, top-right, mid-left, mid-right, bot-left, bot-right])
#   Bit 6 is always set (0xA0 base), except the 0x60–0x7F range which is text.
#   Separated graphics: 0xA0–0xBF → same pattern, drawn with gaps.

# Map 6-bit pattern to the Mode 7 contiguous-graphics byte value.
# Bits: b0=top-left, b1=top-right, b2=mid-left, b3=mid-right, b4=bot-left, b5=bot-right
# The Mode 7 encoding is NOT straightforward — the chip uses a non-linear mapping
# of the 6 bits into the character byte.
#
# Mode 7 gfx chars: 0xA0 (all bg = 0 bits) to 0xBF (all fg = 0b011111 = 31 bits set, skip bit6)
# then 0xE0-0xFF for chars with bit 6 set of the 6-bit pattern.
#
# Actually the mapping is:
#   char_byte = 0xA0 | (pattern & 0x1F) | ((pattern & 0x20) << 1)
# i.e. bit5 of pattern → bit6 of byte, bits 0-4 unchanged.

def pattern_to_gfx_byte(pattern6):
    """Convert 6-bit sub-pixel pattern to Mode 7 graphics character byte."""
    low5 = pattern6 & 0x1F
    bit5 = (pattern6 >> 5) & 1
    return 0xA0 | low5 | (bit5 << 6)


def gfx_byte_to_pattern(byte):
    """Convert Mode 7 graphics character byte to 6-bit sub-pixel pattern."""
    if not (0xA0 <= byte <= 0xBF or 0xE0 <= byte <= 0xFF):
        return None
    low5 = byte & 0x1F
    bit5 = (byte >> 6) & 1
    return low5 | (bit5 << 5)


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

def render_bytes(page_bytes, out_w=640, out_h=480):
    """
    Render a 1000-byte Mode 7 page to an RGB image at the requested dimensions.
    Uses the same fractional sub-pixel grid as the decoder, so comparisons are
    apples-to-apples when out_w/out_h match the source image dimensions.

    Returns a PIL Image.
    """
    px_per_sp_col = out_w / SP_COLS
    px_per_sp_row = out_h / SP_ROWS
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for row in range(N_ROWS):
        cur_fg = 7
        cur_bg = 0
        for col in range(N_COLS):
            b = page_bytes[row * N_COLS + col]

            if 0x91 <= b <= 0x97:
                cur_fg = b - 0x90
                fg_col = cur_bg   # control code cell shows bg
                bg_col = cur_bg
                pat = 0
            elif b == M7_NEW_BG:
                cur_bg = cur_fg
                fg_col = cur_bg
                bg_col = cur_bg
                pat = 0
            elif b == M7_BLACK_BG:
                cur_bg = 0
                fg_col = cur_bg
                bg_col = cur_bg
                pat = 0
            elif 0xA0 <= b <= 0xBF or 0xE0 <= b <= 0xFF:
                pat    = gfx_byte_to_pattern(b)
                fg_col = cur_fg
                bg_col = cur_bg
            else:
                # Space or alpha character — show as background
                pat    = 0
                fg_col = cur_bg
                bg_col = cur_bg

            # Paint the 6 sub-pixels into the output image
            for sr in range(3):
                for sc in range(2):
                    bit = sr * 2 + sc
                    colour = PALETTE_RGB[fg_col if (pat >> bit) & 1 else bg_col].astype(np.uint8)
                    sp_r = row * 3 + sr
                    sp_c = col * 2 + sc
                    py0 = int(round(sp_r * px_per_sp_row))
                    py1 = int(round((sp_r + 1) * px_per_sp_row))
                    px0 = int(round(sp_c * px_per_sp_col))
                    px1 = int(round((sp_c + 1) * px_per_sp_col))
                    out[py0:py1, px0:px1] = colour

    return Image.fromarray(out)


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
