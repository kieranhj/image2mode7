#!/usr/bin/env python3
"""
image2teletext.py - Convert PNG to Teletext/Mode 7 graphics (40x25 characters)

Produces a 1000-byte raw binary file suitable for BBC Micro Mode 7 / Teletext.
Uses graphics colour codes, hold graphics, new background, and sixel block chars.

Usage:
  python image2teletext.py input.png
  python image2teletext.py input.png -o output.bin
  python image2teletext.py input.png --preview out.png
  python image2teletext.py input.png --url
  python image2teletext.py input.png --preview out.png --url
"""

import sys
import argparse
import base64
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

try:
    import numba as _nb
    _NUMBA = True
except ImportError:
    _nb = None
    _NUMBA = False

# ---------------------------------------------------------------------------
# Teletext / Mode 7 constants
# ---------------------------------------------------------------------------

MODE7_WIDTH  = 40
MODE7_HEIGHT = 25
MODE7_MAX_SIZE = MODE7_WIDTH * MODE7_HEIGHT

# Maximum pixel dimensions of the image area
MODE7_PIXEL_W = 78   # (MODE7_WIDTH - 1) * 2
MODE7_PIXEL_H = 75   # MODE7_HEIGHT * 3

# Column 0 of each row is used for the initial graphics colour code;
# image content starts at column 1.
FRAME_FIRST_COLUMN = 1

# Control-code byte values (as stored in the raw .bin)
MODE7_BLANK       = 32   # 0x20 - space / all-off graphics char
MODE7_BLACK_BG    = 156  # 0x9C
MODE7_NEW_BG      = 157  # 0x9D
MODE7_HOLD_GFX    = 158  # 0x9E
MODE7_RELEASE_GFX = 159  # 0x9F
MODE7_GFX_COLOUR  = 144  # 0x90  (add 1-7 for colours red..white)
MODE7_CONTIG_GFX  = 153  # 0x99
MODE7_SEP_GFX     = 154  # 0x9A

SEP_FG_FACTOR = 128  # blending factor for separated graphics (0-255)

# Perceptual luminance weights for RGB error (ITU-R BT.601)
# Human vision is ~6× more sensitive to green than blue; weighting the error
# metric accordingly makes the DP prioritise brightness accuracy over hue.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Lookup table: sRGB byte value (0-255) → linearised value scaled back to 0-255.
# The Teletext palette contains only 0 and 255, which map to 0 and 255 unchanged,
# so only source pixel values need this correction.
# Formula: c = v/255; linear = c/12.92 if c<=0.04045 else ((c+0.055)/1.055)^2.4
_c = np.arange(256, dtype=np.float64) / 255.0
_SRGB_LUT = np.where(_c <= 0.04045, _c / 12.92,
                     ((_c + 0.055) / 1.055) ** 2.4).astype(np.float32) * 255.0
del _c

# State is a 15-bit integer: (sep:1)(last_gfx:7)(hold:1)(bg:3)(fg:3)
MAX_STATE = 1 << 15  # 32768

# Teletext colour palette: index → (R, G, B)
COLOR_RGB = [
    (  0,   0,   0),  # 0 black
    (255,   0,   0),  # 1 red
    (  0, 255,   0),  # 2 green
    (255, 255,   0),  # 3 yellow
    (  0,   0, 255),  # 4 blue
    (255,   0, 255),  # 5 magenta
    (  0, 255, 255),  # 6 cyan
    (255, 255, 255),  # 7 white
]

# Bit-positions within a graphics character for the 6 sub-pixels:
#   [TL TR]   bit 0 (1)   bit 1 (2)
#   [ML MR]   bit 2 (4)   bit 3 (8)
#   [BL BR]   bit 4 (16)  bit 6 (64)   ← bit 5 is always 1 (0x20 base)
GFX_PIXEL_BITS = [1, 2, 4, 8, 16, 64]

# ---------------------------------------------------------------------------
# State encoding (mirrors the C++ GET_STATE macro)
# ---------------------------------------------------------------------------

def pack_state(fg, bg, hold_mode, last_gfx_char, sep):
    return (sep << 14) | (last_gfx_char << 7) | (hold_mode << 6) | (bg << 3) | fg

def unpack_state(state):
    fg           = state & 7
    bg           = (state >> 3) & 7
    hold_mode    = (state >> 6) & 1
    last_gfx     = (state >> 7) & 0x7F
    sep          = (state >> 14) & 1
    return fg, bg, hold_mode, last_gfx, sep

def next_state(proposed_char, old_state, use_hold=True, use_fill=True, use_sep=True):
    """Return the state that results from placing proposed_char in old_state."""
    fg, bg, hold_mode, last_gfx, sep = unpack_state(old_state)

    if use_fill:
        if proposed_char == MODE7_NEW_BG:
            bg = fg
        elif proposed_char == MODE7_BLACK_BG:
            bg = 0

    if MODE7_GFX_COLOUR < proposed_char < MODE7_GFX_COLOUR + 8:
        fg = proposed_char - MODE7_GFX_COLOUR

    if use_hold:
        if proposed_char == MODE7_HOLD_GFX:
            hold_mode = 1
        elif proposed_char == MODE7_RELEASE_GFX:
            hold_mode = 0
            last_gfx = MODE7_BLANK
        elif proposed_char < 128:
            last_gfx = proposed_char
    else:
        hold_mode = 0
        last_gfx = MODE7_BLANK

    if use_sep:
        if proposed_char == MODE7_SEP_GFX:
            sep = 1
        elif proposed_char == MODE7_CONTIG_GFX:
            sep = 0

    return pack_state(fg, bg, hold_mode, last_gfx, sep)

# ---------------------------------------------------------------------------
# Error / pixel rendering
# ---------------------------------------------------------------------------

def screen_rgb(screen_bit, fg, bg, sep):
    """Return (R,G,B) for a single sub-pixel given fg/bg colours and sep mode."""
    if screen_bit:
        if sep:
            fr, fg_g, fb = COLOR_RGB[fg]
            br, bg_g, bb = COLOR_RGB[bg]
            f = SEP_FG_FACTOR
            return (
                (f * fr + (255 - f) * br) // 255,
                (f * fg_g + (255 - f) * bg_g) // 255,
                (f * fb + (255 - f) * bb) // 255,
            )
        return COLOR_RGB[fg]
    return COLOR_RGB[bg]

def pixel_error(img, px, py, screen_bit, fg, bg, sep):
    ir, ig, ib = int(img[py, px, 0]), int(img[py, px, 1]), int(img[py, px, 2])
    sr, sg, sb = screen_rgb(screen_bit, fg, bg, sep)
    return (sr - ir) ** 2 + (sg - ig) ** 2 + (sb - ib) ** 2

def char_error(img, x7, y7, screen_char, fg, bg, sep):
    """Squared-error sum for all 6 sub-pixels of a displayed character cell."""
    x = (x7 - FRAME_FIRST_COLUMN) * 2
    y = y7 * 3
    err  = pixel_error(img, x,   y,   (screen_char)      & 1, fg, bg, sep)
    err += pixel_error(img, x+1, y,   (screen_char >> 1) & 1, fg, bg, sep)
    err += pixel_error(img, x,   y+1, (screen_char >> 2) & 1, fg, bg, sep)
    err += pixel_error(img, x+1, y+1, (screen_char >> 3) & 1, fg, bg, sep)
    err += pixel_error(img, x,   y+2, (screen_char >> 4) & 1, fg, bg, sep)
    err += pixel_error(img, x+1, y+2, (screen_char >> 6) & 1, fg, bg, sep)
    return err

def placed_char_error(img, x7, y7, proposed_char, fg, bg, hold_mode, last_gfx, sep):
    """Error for placing proposed_char at (x7, y7) in the given state."""
    if proposed_char >= 128:
        # Control code: visually shows held char (or blank if hold off)
        screen_char = last_gfx if hold_mode else MODE7_BLANK
    else:
        screen_char = proposed_char
    return char_error(img, x7, y7, screen_char, fg, bg, sep)

def best_gfx_char(img, x7, y7, fg, bg, sep):
    """Find the lowest-error contiguous graphics char given fg/bg colours."""
    x = (x7 - FRAME_FIRST_COLUMN) * 2
    y = y7 * 3
    c = MODE7_BLANK
    for bit, (dx, dy) in zip(GFX_PIXEL_BITS, [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)]):
        px, py = x + dx, y + dy
        if pixel_error(img, px, py, 1, fg, bg, sep) < pixel_error(img, px, py, 0, fg, bg, sep):
            c |= bit
    return c

def _smooth_colour_runs(row, arr, y7, frame_w, min_run,
                        use_hold=True, use_fill=True, use_sep=False):
    """
    Post-process a solved Mode 7 row to merge colour runs shorter than
    min_run cells into the dominant neighbouring colour, then re-render
    affected cells with the new colour via best_gfx_char.

    row[0]             = colour code at col_off
    row[1..frame_w]    = character cells (FRAME_FIRST_COLUMN=1 offset)

    Non-graphics, non-fg-colour control codes (MODE7_NEW_BG, MODE7_BLACK_BG,
    MODE7_HOLD_GFX, etc.) are "immune": they are never replaced and act as
    barriers that prevent run-merging from crossing their position.

    Returns a (possibly new) row list.
    """
    if min_run < 2 or frame_w == 0:
        return row

    # --- 1. Simulate state, record fg/bg/sep active at each cell xi ---
    state = pack_state(1, 0, False, MODE7_BLANK, False)
    state = next_state(row[0], state, use_hold, use_fill, use_sep)

    fg_at  = []
    bg_at  = []
    sep_at = []
    for xi in range(frame_w):
        fg, bg, _, _, sep = unpack_state(state)
        fg_at.append(fg)
        bg_at.append(bg)
        sep_at.append(sep)
        state = next_state(row[FRAME_FIRST_COLUMN + xi], state,
                           use_hold, use_fill, use_sep)

    # Mark positions whose cell is an immune control code (must not be touched).
    # Graphics chars 32-127 and fg colour codes 145-151 can be replaced;
    # everything else (NEW_BG=157, BLACK_BG=156, HOLD_GFX=158, …) is immune.
    immune = [False] * frame_w
    for xi in range(frame_w):
        ch = row[FRAME_FIRST_COLUMN + xi]
        immune[xi] = not (32 <= ch <= 127 or
                          MODE7_GFX_COLOUR < ch < MODE7_GFX_COLOUR + 8)

    orig_fg = fg_at[:]

    # --- 2. Build run-length list [(start, end_exclusive, colour), …]
    #        Immune cells form singleton barrier runs with sentinel colour -1.
    #        Runs immediately preceding a MODE7_NEW_BG barrier get sentinel -2
    #        ("frozen") because MODE7_NEW_BG sets bg = current fg: if we change
    #        the fg of the preceding run the background colour changes for the
    #        rest of the row, causing visible colour bleeding.
    _BARRIER = -1
    _FROZEN  = -2

    def _make_runs(fg, imm):
        runs, i = [], 0
        while i < len(fg):
            if imm[i]:
                runs.append([i, i + 1, _BARRIER])
                i += 1
            else:
                c, j = fg[i], i + 1
                while j < len(fg) and fg[j] == c and not imm[j]:
                    j += 1
                runs.append([i, j, c])
                i = j
        return runs

    runs = _make_runs(fg_at, immune)
    if len(runs) <= 1:
        return row

    # Freeze the run immediately preceding each MODE7_NEW_BG barrier.
    for i, r in enumerate(runs):
        if r[2] == _BARRIER and row[FRAME_FIRST_COLUMN + r[0]] == MODE7_NEW_BG:
            for j in range(i - 1, -1, -1):
                if runs[j][2] not in (_BARRIER, _FROZEN):
                    runs[j][2] = _FROZEN
                    break

    # --- 3. Iteratively merge short runs into the larger neighbour ---
    changed = True
    while changed:
        changed = False
        for i in range(len(runs)):
            if runs[i][2] in (_BARRIER, _FROZEN):
                continue  # barriers and frozen runs are never merged
            rlen = runs[i][1] - runs[i][0]
            if rlen >= min_run:
                continue
            if len(runs) == 1:
                break
            # Find the nearest mergeable neighbours
            left_ok  = i > 0 and runs[i - 1][2] not in (_BARRIER, _FROZEN)
            right_ok = i < len(runs) - 1 and runs[i + 1][2] not in (_BARRIER, _FROZEN)
            if not left_ok and not right_ok:
                continue  # sandwiched between barriers/frozen runs; can't merge
            if not left_ok:
                target = runs[i + 1][2]
            elif not right_ok:
                target = runs[i - 1][2]
            else:
                ll = runs[i - 1][1] - runs[i - 1][0]
                rl = runs[i + 1][1] - runs[i + 1][0]
                target = runs[i - 1][2] if ll >= rl else runs[i + 1][2]
            if target != runs[i][2]:
                for xi in range(runs[i][0], runs[i][1]):
                    fg_at[xi] = target
                runs[i][2] = target
                changed = True
        # Collapse adjacent same-colour runs (barriers and frozen runs stay separate)
        merged = [runs[0][:]]
        for r in runs[1:]:
            if r[2] not in (_BARRIER, _FROZEN) and r[2] == merged[-1][2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r[:])
        runs = merged

    if fg_at == orig_fg:
        return row   # nothing changed

    # --- 4. Re-encode: rebuild cells with updated colour assignments ---
    new_row = list(row)
    new_row[0] = MODE7_GFX_COLOUR + fg_at[0]  # update initial colour code
    cur_fg = fg_at[0]

    for xi in range(frame_w):
        if immune[xi]:
            # Never touch immune control codes; cur_fg is unchanged by them.
            continue

        pos      = FRAME_FIRST_COLUMN + xi
        old_ch   = row[pos]
        new_fg   = fg_at[xi]
        was_ctrl = MODE7_GFX_COLOUR < old_ch < MODE7_GFX_COLOUR + 8

        if new_fg != cur_fg:
            # Colour transition needed here — insert a colour code
            new_row[pos] = MODE7_GFX_COLOUR + new_fg
            cur_fg = new_fg
        elif was_ctrl:
            # This fg colour code appears redundant (new_fg == cur_fg), so we
            # would normally replace it with a graphics char.  However, if the
            # code's own colour (what it SETS for subsequent cells) differs from
            # cur_fg AND the next cell is immune (e.g. MODE7_NEW_BG), we cannot
            # insert a correcting code at xi+1 — the immune cell would read the
            # wrong fg and set the wrong background for the rest of the row.
            # In that case keep the original code and advance cur_fg.
            code_colour = old_ch - MODE7_GFX_COLOUR
            if code_colour != cur_fg and xi + 1 < frame_w and immune[xi + 1]:
                cur_fg = code_colour   # keep original code; new_row[pos] already = old_ch
            else:
                new_row[pos] = best_gfx_char(arr, xi + FRAME_FIRST_COLUMN, y7,
                                             new_fg, bg_at[xi], sep_at[xi])
        elif new_fg != orig_fg[xi]:
            # Colour changed but no code needed — re-render with new colour
            new_row[pos] = best_gfx_char(arr, xi + FRAME_FIRST_COLUMN, y7,
                                         new_fg, bg_at[xi], sep_at[xi])
        # else: colour unchanged, keep original character

    return new_row


_PALETTE_NP = np.array(COLOR_RGB, dtype=np.float32)  # (8, 3)

def snap_to_palette(arr, threshold):
    """
    Snap each pixel to the nearest Teletext palette colour if its Euclidean
    RGB distance to that colour is within `threshold` (0–255 scale).
    Pixels further away than the threshold are left unchanged.

    Applied after resize/quant/sharpen and before dithering, so the ditherer
    only has to handle genuinely ambiguous mid-tones.
    """
    flat = arr.reshape(-1, 3).astype(np.float32)          # (N, 3)
    diffs = flat[:, np.newaxis, :] - _PALETTE_NP           # (N, 8, 3)
    sq_dists = (diffs * diffs).sum(axis=2)                 # (N, 8)
    nearest_idx = sq_dists.argmin(axis=1)                  # (N,)
    nearest_sq = sq_dists[np.arange(len(flat)), nearest_idx]
    mask = nearest_sq <= threshold * threshold             # compare in sq space
    result = flat.copy()
    result[mask] = _PALETTE_NP[nearest_idx[mask]]
    return result.reshape(arr.shape).astype(np.uint8)


def floyd_steinberg_dither(arr):
    """
    Floyd-Steinberg error diffusion dithering, quantizing to the 8-colour
    Teletext palette.  Applied at sub-pixel level after resize and sharpening.

    Each pixel is snapped to the nearest palette colour and the quantization
    error is diffused to the four eastern/southern neighbours with the classic
    7/16, 3/16, 5/16, 1/16 weights.  At Mode 7's low sub-pixel resolution the
    patterns may be visible, but dithering can improve gradients and tone range.

    Input/output: (H, W, 3) uint8 array.
    """
    palette = np.array(COLOR_RGB, dtype=np.float32)   # (8, 3)
    buf = arr[:, :, :3].astype(np.float32)             # work in float to accumulate error
    h, w = buf.shape[:2]

    for y in range(h):
        for x in range(w):
            old = buf[y, x].copy()
            dists = ((old[None, :] - palette) ** 2).sum(axis=1)
            new   = palette[dists.argmin()]
            buf[y, x] = new
            err = old - new                            # quantization error (3,)

            if x + 1 < w:
                buf[y,     x + 1] = np.clip(buf[y,     x + 1] + err * (7 / 16), 0, 255)
            if y + 1 < h:
                if x > 0:
                    buf[y + 1, x - 1] = np.clip(buf[y + 1, x - 1] + err * (3 / 16), 0, 255)
                buf[y + 1, x    ] = np.clip(buf[y + 1, x    ] + err * (5 / 16), 0, 255)
                if x + 1 < w:
                    buf[y + 1, x + 1] = np.clip(buf[y + 1, x + 1] + err * (1 / 16), 0, 255)

    return buf.astype(np.uint8)


# ---------------------------------------------------------------------------
# Precompute per-row error lookup table (numpy vectorised)
# ---------------------------------------------------------------------------

def build_error_table(img, y7, frame_w, luma=False, linear=False):
    """
    Return err_table[x7_idx, fg, bg, sep, screen_char] = total squared RGB error
    for displaying screen_char (0-127) at character column x7_idx in row y7,
    with the given fg/bg palette indices and sep flag.

    Also returns gfx_table[x7_idx, fg, bg, sep] = optimal graphics character.

    linear: if True, linearise source pixel values from sRGB to linear light
            before computing squared error (more perceptually correct).
            The Teletext palette is all 0s and 255s so is unchanged by linearisation.
    """
    # Extract pixel values: shape (frame_w, 6, 3)
    pixels = np.empty((frame_w, 6, 3), dtype=np.float32)
    for xi in range(frame_w):
        x = xi * 2
        y = y7 * 3
        pixels[xi, 0] = img[y,   x,   :3]
        pixels[xi, 1] = img[y,   x+1, :3]
        pixels[xi, 2] = img[y+1, x,   :3]
        pixels[xi, 3] = img[y+1, x+1, :3]
        pixels[xi, 4] = img[y+2, x,   :3]
        pixels[xi, 5] = img[y+2, x+1, :3]

    if linear:
        # Map each uint8 pixel value through the sRGB→linear LUT.
        # Palette colours (0 and 255) are unchanged (0→0, 255→255), so only
        # source pixels need correction; disp computation below is unaffected.
        pixels = _SRGB_LUT[pixels.astype(np.int32)]

    # screen_bits[sc, sixel] = 1 if that sub-pixel is "on" for screen char sc
    # Bit positions: [0,1,2,3,4,6] map to sixels [TL,TR,ML,MR,BL,BR]
    sc_arr = np.arange(128, dtype=np.int32)
    screen_bits = np.column_stack([
        (sc_arr >> 0) & 1,  # bit 0 → TL
        (sc_arr >> 1) & 1,  # bit 1 → TR
        (sc_arr >> 2) & 1,  # bit 2 → ML
        (sc_arr >> 3) & 1,  # bit 3 → MR
        (sc_arr >> 4) & 1,  # bit 4 → BL
        (sc_arr >> 6) & 1,  # bit 6 → BR
    ]).astype(np.float32)  # (128, 6)

    palette = np.array(COLOR_RGB, dtype=np.float32)  # (8, 3)
    f = SEP_FG_FACTOR / 255.0

    err_table = np.empty((frame_w, 8, 8, 2, 128), dtype=np.int32)
    gfx_table = np.empty((frame_w, 8, 8, 2), dtype=np.uint8)

    for sep in range(2):
        for fg in range(8):
            for bg in range(8):
                fg_c = palette[fg]
                bg_c = palette[bg]
                if sep:
                    fg_disp = f * fg_c + (1 - f) * bg_c
                else:
                    fg_disp = fg_c

                # display_rgb[sc, sixel, ch] = bg_c + bit * (fg_disp - bg_c)
                # screen_bits: (128, 6), fg_disp/bg_c: (3,)
                diff = fg_disp - bg_c  # (3,)
                disp = bg_c[None, None, :] + screen_bits[:, :, None] * diff[None, None, :]
                # disp: (128, 6, 3)

                # squared error vs pixels[xi]: (frame_w, 128, 6, 3)
                sq = (disp[None, :, :, :] - pixels[:, None, :, :]) ** 2
                weights = LUMA_WEIGHTS if luma else np.ones(3, dtype=np.float32)
                err_table[:, fg, bg, sep, :] = (sq * weights).sum(axis=(2, 3)).astype(np.int32)

                # Best graphics char: each sub-pixel independently on or off
                # on_err[xi, sixel] = err_table[xi, fg, bg, sep, sc] where only that bit set
                # off_err = err_table[...] with that bit clear
                # Determine bit-by-bit which minimises error
                c_arr = np.full(frame_w, MODE7_BLANK, dtype=np.uint8)
                for bit_idx, bit_val in enumerate(GFX_PIXEL_BITS):
                    sc_on  = np.uint8(MODE7_BLANK | bit_val)
                    sc_off = np.uint8(MODE7_BLANK)
                    # Compare marginal error: full char with just this bit toggled
                    on_e  = err_table[:, fg, bg, sep, sc_on]
                    off_e = err_table[:, fg, bg, sep, sc_off]
                    # For each xi, if on is better set the bit
                    c_arr = np.where(on_e < off_e, c_arr | np.uint8(bit_val), c_arr)
                gfx_table[:, fg, bg, sep] = c_arr

    return err_table, gfx_table


def _candidates(state, use_hold, use_fill, use_sep, gfx_char):
    """Return list of candidate byte values to try at a given state."""
    fg, bg, hold_mode, last_gfx, sep = unpack_state(state)
    cands = [MODE7_BLANK]
    if gfx_char != MODE7_BLANK:
        cands.append(gfx_char)
    for c in range(1, 8):
        if c != fg:
            cands.append(MODE7_GFX_COLOUR + c)
    if use_fill:
        if bg != fg:
            cands.append(MODE7_NEW_BG)
        if bg != 0:
            cands.append(MODE7_BLACK_BG)
    if use_hold:
        cands.append(MODE7_HOLD_GFX if not hold_mode else MODE7_RELEASE_GFX)
    if use_sep:
        cands.append(MODE7_SEP_GFX if not sep else MODE7_CONTIG_GFX)
    return cands


def _effective_cell(proposed_char, fg, bg, hold_mode, last_gfx, sep):
    """
    Return (display_char, eff_bg) for error computation, applying Set-At effects.
    NEW_BG / BLACK_BG / HOLD_GFX / RELEASE_GFX take effect at the cell that
    contains them (Set-At), so the error must be computed with the new state.
    fg colour changes and SEP codes are Set-After (take effect next cell).
    """
    eff_bg   = bg
    eff_hold = hold_mode
    eff_last = last_gfx
    if proposed_char == MODE7_NEW_BG:
        eff_bg = fg
    elif proposed_char == MODE7_BLACK_BG:
        eff_bg = 0
    elif proposed_char == MODE7_HOLD_GFX:
        eff_hold = True
    elif proposed_char == MODE7_RELEASE_GFX:
        eff_hold = False
        eff_last = MODE7_BLANK
    if proposed_char >= 128:
        dc = eff_last if eff_hold else MODE7_BLANK
    else:
        dc = proposed_char
    return dc, eff_bg


def _greedy_row_from(start_fg, err_table, gfx_table, frame_w,
                     use_hold, use_fill, use_sep):
    """
    Run the greedy solver (2-step lookahead) from a given starting colour.
    Returns (result, total_error) where total_error is the sum of immediate
    per-character errors (used to compare starting colours).
    """
    row_end = FRAME_FIRST_COLUMN + frame_w
    result = [MODE7_BLANK] * MODE7_WIDTH
    result[0] = MODE7_GFX_COLOUR + start_fg
    state = pack_state(start_fg, 0, False, MODE7_BLANK, False)
    total_err = 0

    for x7 in range(FRAME_FIRST_COLUMN, row_end):
        xi = x7 - FRAME_FIRST_COLUMN
        fg, bg, hold_mode, last_gfx, sep = unpack_state(state)
        gc = int(gfx_table[xi, fg, bg, sep])
        cands = _candidates(state, use_hold, use_fill, use_sep, gc)

        best_err  = 10 ** 18
        best_char = MODE7_BLANK

        for char in cands:
            dc, eff_bg = _effective_cell(char, fg, bg, hold_mode, last_gfx, sep)
            err = int(err_table[xi, fg, eff_bg, sep, dc])

            # One-step lookahead
            if x7 + 1 < row_end:
                ns = next_state(char, state, use_hold, use_fill, use_sep)
                nfg, nbg, nhold, nlast, nsep = unpack_state(ns)
                ngc = int(gfx_table[xi + 1, nfg, nbg, nsep])
                ndc, neff_bg = _effective_cell(ngc, nfg, nbg, nhold, nlast, nsep)
                err += int(err_table[xi + 1, nfg, neff_bg, nsep, ndc])

            if err < best_err:
                best_err  = err
                best_char = char

        result[x7] = best_char
        # Score by immediate error only (not lookahead) for fair start comparison
        dc, eff_bg = _effective_cell(best_char, fg, bg, hold_mode, last_gfx, sep)
        total_err += int(err_table[xi, fg, eff_bg, sep, dc])
        state = next_state(best_char, state, use_hold, use_fill, use_sep)

    return result, total_err


# ---------------------------------------------------------------------------
# Fast greedy solver (--greedy — multi-start, 2-step lookahead)
# ---------------------------------------------------------------------------

def greedy_row(err_table, gfx_table, frame_w,
               use_hold=True, use_fill=True, use_sep=False):
    """
    Greedy row encoder with 2-step lookahead and multi-start.
    Tries all 7 starting colours and keeps the lowest-error result.
    """
    best_result = None
    best_err    = 10 ** 18
    for fg in range(1, 8):
        result, err = _greedy_row_from(fg, err_table, gfx_table, frame_w,
                                       use_hold, use_fill, use_sep)
        if err < best_err:
            best_err    = err
            best_result = result
    return best_result


# ---------------------------------------------------------------------------
# Local-search refinement pass (--refine)
# ---------------------------------------------------------------------------

def _tail_error(result, start_xi, state, err_table, frame_w, use_hold, use_fill, use_sep):
    """
    Sum of immediate per-character errors from frame position start_xi to end,
    given the decoder state at start_xi.  Characters in result[] are held fixed;
    state propagation is re-simulated from the supplied state.
    """
    total = 0
    for xi in range(start_xi, frame_w):
        x7 = xi + FRAME_FIRST_COLUMN
        char = result[x7]
        fg, bg, hold_mode, last_gfx, sep = unpack_state(state)
        dc, eff_bg = _effective_cell(char, fg, bg, hold_mode, last_gfx, sep)
        total += int(err_table[xi, fg, eff_bg, sep, dc])
        state = int(next_state(char, state, use_hold, use_fill, use_sep))
    return total


def refine_row(result, err_table, gfx_table, frame_w,
               use_hold=True, use_fill=True, use_sep=False):
    """
    Local-search refinement pass over a row produced by any solver.

    Sweeps left-to-right; at each position tries every valid candidate
    character.  For each candidate the cost is:
        error_at_xi  +  tail_error(xi+1, new_state, fixed_chars)
    where the remaining characters are kept fixed but the decoder state is
    re-simulated from xi+1 with the new state.  The substitution is accepted
    if it strictly lowers the combined cost.

    Passes repeat until no position improves (convergence — typically 1-3
    passes in practice).  This can improve DP output because fixing subsequent
    chars in a new state sometimes yields a lower combined cost than the DP's
    assumption of independent optimal future choices.
    """
    result = list(result)   # work on a mutable copy

    improved = True
    while improved:
        improved = False

        # Recompute decoder state entering each frame position.
        init_fg  = result[0] - MODE7_GFX_COLOUR
        state    = pack_state(init_fg, 0, False, MODE7_BLANK, False)
        states   = [0] * (frame_w + 1)
        states[0] = state
        for xi in range(frame_w):
            x7    = xi + FRAME_FIRST_COLUMN
            state = int(next_state(result[x7], state, use_hold, use_fill, use_sep))
            states[xi + 1] = state

        for xi in range(frame_w):
            x7    = xi + FRAME_FIRST_COLUMN
            state = states[xi]
            fg, bg, hold_mode, last_gfx, sep = unpack_state(state)
            gc    = int(gfx_table[xi, fg, bg, sep])
            cands = _candidates(state, use_hold, use_fill, use_sep, gc)

            cur_char             = result[x7]
            cur_dc, cur_eff_bg   = _effective_cell(cur_char, fg, bg, hold_mode, last_gfx, sep)
            cur_err              = int(err_table[xi, fg, cur_eff_bg, sep, cur_dc])
            cur_ns               = int(next_state(cur_char, state, use_hold, use_fill, use_sep))
            cur_tail             = _tail_error(result, xi + 1, cur_ns,
                                               err_table, frame_w, use_hold, use_fill, use_sep)
            best_total           = cur_err + cur_tail
            best_char            = cur_char

            for char in cands:
                if char == cur_char:
                    continue
                dc, eff_bg = _effective_cell(char, fg, bg, hold_mode, last_gfx, sep)
                err  = int(err_table[xi, fg, eff_bg, sep, dc])
                ns   = int(next_state(char, state, use_hold, use_fill, use_sep))
                tail = _tail_error(result, xi + 1, ns,
                                   err_table, frame_w, use_hold, use_fill, use_sep)
                if err + tail < best_total:
                    best_total = err + tail
                    best_char  = char

            if best_char != cur_char:
                result[x7] = best_char
                improved   = True
                # Update precomputed states from xi+1 onwards for this pass
                ns = int(next_state(best_char, states[xi], use_hold, use_fill, use_sep))
                for j in range(xi + 1, frame_w + 1):
                    states[j] = ns
                    if j < frame_w:
                        ns = int(next_state(result[j + FRAME_FIRST_COLUMN], ns,
                                            use_hold, use_fill, use_sep))

    return result


# ---------------------------------------------------------------------------
# Numba-accelerated DP column kernel (used by dp_row when numba is installed)
# ---------------------------------------------------------------------------
# Arrays are transposed vs the numpy path so the inner s-loop accesses
# contiguous memory: ns_gfx_T[s, j] and ns_ctrl_T[s, i] are sequential.
# The first call triggers JIT compilation (~2-3 s); subsequent calls are fast
# and the compiled code is cached to disk (cache=True).

if _NUMBA:
    @_nb.njit(parallel=False, cache=True, nogil=True)
    def _dp_col_nb(et_col,              # (128, 128) int32 — error table for this column
                   fg_bg_sep_idx,       # (N,) int32
                   fg_bg_sep_fg_idx,    # (N,) int32
                   fg_bg_sep_zero_idx,  # (N,) int32
                   ctrl_disp,           # (N,) int32
                   last_v,              # (N,) int32
                   ns_ctrl_T,           # (N, NC) int32 — ctrl next-states, transposed
                   char_ctrl_vals,      # (NC,) int32 — output char per ctrl cand
                   mask_ctrl_T,         # (N, NC) bool — validity, transposed
                   err_codes,           # (NC,) int32 — 0/1/2/4/5 display selector
                   gfx_chars,           # (G,) int32 — the 63 gfx char values
                   ns_gfx_T,            # (N, G) int32 — gfx next-states, transposed
                   dp_in,               # (N,) int32 — cost-to-go entering this col
                   dp_out,              # (N,) int32 — cost-to-go (output)
                   bc_out,              # (N,) int32 — best char (output)
                   INF):
        NC    = char_ctrl_vals.shape[0]
        NUM_G = gfx_chars.shape[0]
        BLANK = 32
        for s in range(dp_in.shape[0]):
            fbs    = fg_bg_sep_idx[s]
            fbs_fg = fg_bg_sep_fg_idx[s]
            fbs_z  = fg_bg_sep_zero_idx[s]
            cd     = ctrl_disp[s]
            lv     = last_v[s]
            # Six possible display-error base values, selected by err_code
            e0 = et_col[fbs,    BLANK]
            e1 = et_col[fbs,    cd]
            e2 = et_col[fbs,    lv]
            e4 = et_col[fbs_fg, cd]
            e5 = et_col[fbs_z,  cd]
            best_cost = INF
            best_char = BLANK
            # Control candidates (first → wins ties with gfx)
            for i in range(NC):
                if not mask_ctrl_T[s, i]:
                    continue
                ec = err_codes[i]
                if   ec == 0: err = e0
                elif ec == 1: err = e1
                elif ec == 2: err = e2
                elif ec == 4: err = e4
                else:         err = e5
                cost = err + dp_in[ns_ctrl_T[s, i]]
                if cost < best_cost:
                    best_cost = cost
                    best_char = char_ctrl_vals[i]
            # Graphics candidates (63 chars; transposed layout → sequential reads)
            for j in range(NUM_G):
                gc   = gfx_chars[j]
                err  = et_col[fbs, gc]
                cost = err + dp_in[ns_gfx_T[s, j]]
                if cost < best_cost:
                    best_cost = cost
                    best_char = gc
            dp_out[s] = best_cost
            bc_out[s] = best_char


# ---------------------------------------------------------------------------
# Full DP solver (default — near-optimal quality)
# ---------------------------------------------------------------------------

def dp_row(err_table, gfx_table, frame_w,
           use_hold=True, use_fill=True, use_sep=False):
    """
    Vectorised bottom-up DP row encoder.  Matches C++ -slow quality:
    tries all 63 non-blank graphics characters per state so the DP can
    account for HOLD_GFX picking up a non-locally-optimal gfx char later.
    """
    # sep bit is always 0 when use_sep=False, so only the lower half of the
    # state space is reachable — halves every array and nearly every numpy op.
    N_STATES = MAX_STATE if use_sep else MAX_STATE >> 1
    all_s  = np.arange(N_STATES, dtype=np.int32)
    fg_v   = all_s & 7
    bg_v   = (all_s >> 3) & 7
    hold_v = (all_s >> 6) & 1
    last_v = (all_s >> 7) & 0x7F
    sep_v  = (all_s >> 14) & 1
    bg_zero = np.zeros(N_STATES, dtype=np.int32)
    INF = np.int32(2 ** 30)   # larger than any real cumulative error (max ~33M per row)

    # ------------------------------------------------------------------
    # Precompute state-component arrays and column-independent next-states
    # ------------------------------------------------------------------
    ctrl_disp = np.where(hold_v, last_v, np.int32(MODE7_BLANK)).astype(np.int32)
    sep_cur   = (sep_v << 14).astype(np.int32)

    if use_hold:
        hl_blank   = ((np.int32(MODE7_BLANK) << 7) | (hold_v << 6)).astype(np.int32)
        hl_ctrl    = ((last_v << 7) | (hold_v << 6)).astype(np.int32)
        hl_hold    = ((last_v << 7) | np.int32(1 << 6)).astype(np.int32)
        hl_release = np.full(N_STATES, np.int32(MODE7_BLANK << 7), dtype=np.int32)
    else:
        _hl = np.full(N_STATES, np.int32(MODE7_BLANK << 7), dtype=np.int32)
        hl_blank = hl_ctrl = hl_hold = hl_release = _hl

    # All 63 non-blank graphics characters (bit5 maps to bit6 for BR pixel)
    GFX_CHARS = np.array([MODE7_BLANK | (i & 0x1f) | ((i & 0x20) << 1)
                          for i in range(1, 64)], dtype=np.int32)  # (63,)
    NUM_GFX = len(GFX_CHARS)

    # Next-state for each gfx char × each state (precomputed before column loop)
    base_ns = (sep_cur | (hold_v << 6) | (bg_v << 3) | fg_v).astype(np.int32)  # (N_STATES,)
    if use_hold:
        ns_gfx = base_ns[None, :] | (GFX_CHARS[:, None] << 7)  # (63, N_STATES)
    else:
        ns_gfx = np.tile(base_ns | np.int32(MODE7_BLANK << 7), (NUM_GFX, 1))

    # ------------------------------------------------------------------
    # Control-code candidates: (output_char, next_state_array, validity_mask)
    # err_code key:
    #   0 = blank display                    1 = ctrl_disp (current bg)
    #   2 = last_v (HOLD Set-At)             4 = ctrl_disp with bg=fg (NEW_BG)
    #   5 = ctrl_disp with bg=0 (BLACK_BG)
    # ------------------------------------------------------------------
    # Candidate order matches C++ get_error_for_remainder_of_line evaluation order,
    # which determines tie-breaking (strict < means first candidate wins a tie):
    #   BLANK, NEW_BG, BLACK_BG, SEP, HOLD/RELEASE, colours 1-7, gfx chars
    ctrl_cands = []
    ctrl_cands.append((MODE7_BLANK,
                       (sep_cur | hl_blank | (bg_v << 3) | fg_v).astype(np.int32),
                       None, 0))
    if use_fill:
        ctrl_cands.append((MODE7_NEW_BG,
                           (sep_cur | hl_ctrl | (fg_v << 3) | fg_v).astype(np.int32),
                           bg_v != fg_v, 4))
        ctrl_cands.append((MODE7_BLACK_BG,
                           (sep_cur | hl_ctrl | fg_v).astype(np.int32),
                           bg_v != 0, 5))
    if use_sep:
        ctrl_cands.append((MODE7_SEP_GFX,
                           (np.int32(1 << 14) | hl_ctrl | (bg_v << 3) | fg_v).astype(np.int32),
                           sep_v == 0, 1))
        ctrl_cands.append((MODE7_CONTIG_GFX,
                           (hl_ctrl | (bg_v << 3) | fg_v).astype(np.int32),
                           sep_v == 1, 1))
    if use_hold:
        ctrl_cands.append((MODE7_HOLD_GFX,
                           (sep_cur | hl_hold    | (bg_v << 3) | fg_v).astype(np.int32),
                           hold_v == 0, 2))
        ctrl_cands.append((MODE7_RELEASE_GFX,
                           (sep_cur | hl_release | (bg_v << 3) | fg_v).astype(np.int32),
                           hold_v == 1, 0))
    for c in range(1, 8):
        ctrl_cands.append((MODE7_GFX_COLOUR + c,
                           (sep_cur | hl_ctrl | (bg_v << 3) | np.int32(c)).astype(np.int32),
                           fg_v != c, 1))

    NC = len(ctrl_cands)
    num_cands = NC + NUM_GFX
    err_codes_ctrl = [ec for _, _, _, ec in ctrl_cands]

    # ------------------------------------------------------------------
    # Ctrl candidate matrices (NC rows only; gfx handled separately below)
    # ------------------------------------------------------------------
    ns_stack   = np.empty((NC + 1, N_STATES), dtype=np.int32)  # +1 for gfx slot
    char_stack = np.empty((NC + 1, N_STATES), dtype=np.int32)
    mask_stack = np.ones ((NC + 1, N_STATES), dtype=bool)

    for i, (ch, ns, msk, _) in enumerate(ctrl_cands):
        ns_stack[i]   = ns
        char_stack[i] = ch
        if msk is not None:
            mask_stack[i] = msk
    # Row NC is the gfx slot: char updated per column, ns not used (cost pre-computed)

    # Precomputed index arrays for fast per-column lookups
    # err_table[xi] shape (8,8,2,128) → C-order flat first-3 dim: fg*16 + bg*2 + sep
    fg_bg_sep_idx      = (fg_v * 16 + bg_v * 2 + sep_v).astype(np.int32)  # (N_STATES,)
    fg_bg_sep_fg_idx   = (fg_v * 16 + fg_v * 2 + sep_v).astype(np.int32)  # bg=fg
    fg_bg_sep_zero_idx = (fg_v * 16 +           sep_v ).astype(np.int32)   # bg=0
    # For gfx future-cost lookups: dp_next reshaped as (128, 128) [no-sep] or (2,128,128) [sep]
    # sep_base_low_v indexes the outer dimension: 0..127 (no-sep) or 0..255 (sep)
    base_lower_v    = (all_s & np.int32(0x7F)).astype(np.int32)            # (N_STATES,)
    sep_base_low_v  = (sep_v * np.int32(128) + base_lower_v).astype(np.int32)

    best_char_arr = np.full((frame_w, N_STATES), MODE7_BLANK, dtype=np.int32)

    if _NUMBA:
        # ------------------------------------------------------------------
        # Numba path: fused per-state kernel, no large intermediate arrays.
        # Transpose to (N_STATES, ...) so the inner j/i loops are sequential.
        # First call triggers JIT compilation (~2-3 s, cached to disk).
        # ------------------------------------------------------------------
        char_ctrl_vals = np.array([ch for ch, _, _, _ in ctrl_cands], dtype=np.int32)
        err_codes_arr  = np.array(err_codes_ctrl, dtype=np.int32)
        ns_ctrl_T   = np.ascontiguousarray(ns_stack[:NC].T)   # (N_STATES, NC)
        mask_ctrl_T = np.ascontiguousarray(mask_stack[:NC].T)  # (N_STATES, NC)
        ns_gfx_T    = np.ascontiguousarray(ns_gfx.T)          # (N_STATES, NUM_GFX)
        dp_a  = np.zeros(N_STATES, dtype=np.int32)
        dp_b  = np.zeros(N_STATES, dtype=np.int32)
        INF32 = np.int32(2 ** 30)
        for xi in range(frame_w - 1, -1, -1):
            _dp_col_nb(err_table[xi].reshape(128, 128),
                       fg_bg_sep_idx, fg_bg_sep_fg_idx, fg_bg_sep_zero_idx,
                       ctrl_disp, last_v, ns_ctrl_T, char_ctrl_vals,
                       mask_ctrl_T, err_codes_arr, GFX_CHARS, ns_gfx_T,
                       dp_a, dp_b, best_char_arr[xi], INF32)
            dp_a, dp_b = dp_b, dp_a
        dp_next = dp_a

    else:
        # ------------------------------------------------------------------
        # Numpy path — vectorised over all states simultaneously
        # ------------------------------------------------------------------
        dp_next   = np.zeros(N_STATES, dtype=np.int32)
        err_mat   = np.empty((NC,     N_STATES), dtype=np.int32)
        total_mat = np.empty((NC + 1, N_STATES), dtype=np.int32)
        gfx_work  = np.empty((N_STATES, NUM_GFX), dtype=np.int32)

        for xi in range(frame_w - 1, -1, -1):

            et_flat = err_table[xi].reshape(128, 128)       # (fg_bg_sep, char) view, free

            # ----------------------------------------------------------
            # Gfx candidates — two-level to stay cache-friendly
            # Level 1: find best gfx char per state via (N_STATES, 63) argmin axis=1
            #   source tables: (128, 63) err compact + compact dp_next slice → fit in cache
            # ----------------------------------------------------------
            # Error for each gfx char per state: gather from (128, 63) compact table
            gfx_err_compact = et_flat[:, GFX_CHARS]         # (128, 63) — 32KB, L1 friendly
            np.take(gfx_err_compact, fg_bg_sep_idx,
                    axis=0, out=gfx_work)                   # (N_STATES, 63) gather from 32KB

            # dp_next future-cost for gfx next-states.
            # Build a compact table indexed by sep_base_low_v, sliced to NUM_GFX columns.
            # use_sep=True:  dp_next (32768,) → reshape (2,128,128); table is (256, 63)
            # use_sep=False: dp_next (16384,) → reshape  (128,128);  table is (128, 63)
            if use_hold:
                if use_sep:
                    dp_gfx_fut = dp_next.reshape(2, 128, 128)[:, GFX_CHARS, :].transpose(0, 2, 1).reshape(256, NUM_GFX)
                else:
                    dp_gfx_fut = dp_next.reshape(128, 128)[GFX_CHARS, :].T  # (128, NUM_GFX)
                gfx_work += dp_gfx_fut[sep_base_low_v]
            else:
                if use_sep:
                    fut_1d = dp_next.reshape(2, 128, 128)[sep_v, MODE7_BLANK, base_lower_v]
                else:
                    fut_1d = dp_next.reshape(128, 128)[MODE7_BLANK, base_lower_v]
                gfx_work += fut_1d[:, None]

            # argmin axis=1: cache-friendly (63 contiguous values per state)
            best_gfx_j    = gfx_work.argmin(axis=1)         # (N_STATES,)
            best_gfx_cost = gfx_work[all_s, best_gfx_j]     # (N_STATES,)
            char_stack[NC] = GFX_CHARS[best_gfx_j]          # update gfx char slot

            # ----------------------------------------------------------
            # Control candidates — (NC, N_STATES)
            # ----------------------------------------------------------
            ec0 = et_flat[fg_bg_sep_idx,      MODE7_BLANK]   # blank display
            ec1 = et_flat[fg_bg_sep_idx,      ctrl_disp]     # ctrl_disp, current bg
            ec2 = et_flat[fg_bg_sep_idx,      last_v]        # last_gfx (HOLD Set-At)
            if use_fill:
                ec4 = et_flat[fg_bg_sep_fg_idx,   ctrl_disp] # NEW_BG Set-At
                ec5 = et_flat[fg_bg_sep_zero_idx, ctrl_disp] # BLACK_BG Set-At
            else:
                ec4 = ec5 = ec0

            ec_lut = (ec0, ec1, ec2, None, ec4, ec5)
            for i, code in enumerate(err_codes_ctrl):
                err_mat[i] = ec_lut[code]
            np.add(err_mat, dp_next[ns_stack[:NC]], out=total_mat[:NC])
            total_mat[:NC][~mask_stack[:NC]] = INF

            # ----------------------------------------------------------
            # Combine: ctrl (NC rows) + best-gfx (1 row) → argmin over NC+1
            # ----------------------------------------------------------
            total_mat[NC] = best_gfx_cost

            best_c = np.argmin(total_mat, axis=0)            # (N_STATES,) over NC+1 rows
            best_char_arr[xi] = char_stack[best_c, all_s]
            dp_next = total_mat[best_c, all_s]

    # Forward simulation from the initial state to recover the optimal sequence.
    # dp_next[state] now holds the total row cost achievable from that state,
    # so we read off the best starting colour directly from the DP result.
    min_col    = min(range(1, 8),
                     key=lambda fg: int(dp_next[pack_state(fg, 0, False, MODE7_BLANK, False)]))
    result     = [MODE7_BLANK] * MODE7_WIDTH
    result[0]  = MODE7_GFX_COLOUR + min_col
    init_state = pack_state(min_col, 0, False, MODE7_BLANK, False)

    state = init_state
    for xi in range(frame_w):
        x7 = xi + FRAME_FIRST_COLUMN
        char = int(best_char_arr[xi, state])
        result[x7] = char
        state = int(next_state(char, state, use_hold, use_fill, use_sep))

    return result

# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------

_FILTER_MAP = {
    'bilinear': Image.BILINEAR,
    'lanczos':  Image.LANCZOS,
    'bicubic':  Image.BICUBIC,
    'nearest':  Image.NEAREST,
}


def _cimg_nearest_resize(arr, dst_w, dst_h):
    """
    Resize numpy array (H, W, 3) using CImg's default nearest-neighbour interpolation.

    CImg (interpolation_type=1) maps:  source_x = floor(x * src_w / dst_w)
    PIL nearest maps:                  source_x = floor((x + 0.5) * src_w / dst_w)

    The left-aligned (no +0.5) formula matches the offset-table logic in CImg's
    get_resize() case 1, producing an exact pixel-for-pixel match to the C++ exe output.
    """
    src_h, src_w = arr.shape[:2]
    xs = np.floor(np.arange(dst_w) * src_w / dst_w).astype(np.int32)
    ys = np.floor(np.arange(dst_h) * src_h / dst_h).astype(np.int32)
    return arr[np.ix_(ys, xs)]


def preprocess_image(img_path, filter='bilinear', par=1.2,
                     sharpen_radius=1.0, sharpen_amount=0, sharpen_threshold=0,
                     gamma=1.0, contrast=1.0, saturation=1.0, dither=False,
                     quant_colors=0, posterize=0, median=0, snap=0,
                     snap_palette=False):
    """
    Apply tone/colour adjustments, resize to sub-pixel canvas, and sharpening —
    the same pipeline steps that precede the DP solver in convert_image.
    Returns the processed image as a PIL Image at sub-pixel resolution (≤78×75 px).
    Useful for previewing the effect of parameters before running the full conversion.
    """
    from PIL import ImageEnhance
    img = Image.open(img_path).convert('RGB')
    iw, ih = img.size

    if gamma != 1.0:
        arr16 = np.array(img, dtype=np.float32) / 255.0
        arr16 = np.clip(arr16 ** (1.0 / gamma), 0.0, 1.0)
        img = Image.fromarray((arr16 * 255.0).astype(np.uint8))
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if posterize > 0:
        # PIL's ImageOps.posterize produces values in steps of 256/2^bits
        # (e.g. bits=1 → {0, 128}), never reaching 255.  When dithering then
        # tries to represent 128 via the Teletext palette (channels only 0/255)
        # it alternates between them, producing visible black speckles.
        # Instead, quantise to 2^bits levels spaced evenly across 0–255 so the
        # top level always maps to 255.
        levels = 2 ** posterize
        arr_p = np.array(img, dtype=np.float32)
        arr_p = np.round(arr_p / 255.0 * (levels - 1))   # 0 .. levels-1
        arr_p = np.round(arr_p * 255.0 / (levels - 1))   # 0 .. 255
        img = Image.fromarray(arr_p.astype(np.uint8))

    if median > 0:
        from PIL.ImageFilter import MedianFilter
        # median is a radius: size = 2*radius+1 (so 1→3×3, 2→5×5, …)
        img = img.filter(MedianFilter(size=median * 2 + 1))

    # Save the full-size image before resize for palette computation (see below).
    img_full = img

    pw = MODE7_PIXEL_W
    ph = int(round(pw * ih / iw * par))
    if ph % 3:
        ph += 3 - (ph % 3)
    if ph > MODE7_PIXEL_H:
        ph = MODE7_PIXEL_H
        pw = int(round(ph * iw / ih / par))
        if pw % 2:
            pw += 1

    if filter == 'cimg':
        arr = _cimg_nearest_resize(np.array(img, dtype=np.uint8), pw, ph)
    else:
        img = img.resize((pw, ph), _FILTER_MAP[filter])
        arr = np.array(img, dtype=np.uint8)

    if quant_colors > 0:
        # Find a visually diverse palette from the full-size image, then
        # apply it to the resized arr (no dithering — let our own pass handle it).
        #
        # Plain FASTOCTREE/MEDIANCUT on quant_colors is frequency-biased: if the
        # background dominates (e.g. a large sky or water region), all N palette
        # slots go to shades of the dominant colour and the subject's vivid tones
        # are never represented.
        #
        # Two-pass diverse-palette approach:
        #   1. Quantize to N*8 candidates with FASTOCTREE (more colours = better
        #      coverage of the full colour space).
        #   2. Greedily select N of those candidates that maximise pairwise distance
        #      (greedy max-min / "furthest first"), ensuring the palette spans the
        #      image's colour range rather than clustering around the majority hue.
        candidates_n = quant_colors * 8
        cand_p = img_full.quantize(
            colors=candidates_n, method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.NONE)
        cand_rgb = np.array(
            cand_p.getpalette()[:candidates_n * 3],
            dtype=np.float32).reshape(candidates_n, 3)
        # Greedy furthest-first selection
        selected = [0]
        min_dists = np.full(candidates_n, np.inf)
        for _ in range(quant_colors - 1):
            d = np.sum((cand_rgb - cand_rgb[selected[-1]]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d)
            min_dists[selected] = -1
            selected.append(int(np.argmax(min_dists)))
        chosen = cand_rgb[selected].astype(np.uint8)  # (quant_colors, 3)
        # Build a PIL P-mode palette image from the chosen colours and apply it
        palette_data = np.zeros(768, dtype=np.uint8)
        palette_data[:quant_colors * 3] = chosen.flatten()
        palette_img = Image.new('P', (1, 1))
        palette_img.putpalette(palette_data.tolist())
        arr = np.array(
            Image.fromarray(arr)
                 .quantize(palette=palette_img, dither=Image.Dither.NONE)
                 .convert('RGB'),
            dtype=np.uint8)
        if snap_palette:
            # Snap each quantized colour region to its nearest Teletext colour.
            # After quantization every pixel is one of N palette centroids, so
            # snapping with a huge threshold maps each centroid unconditionally.
            # The result uses only Teletext colours but with the smooth region
            # boundaries that quantization provides — the "art palette" look.
            arr = snap_to_palette(arr, 99999)
    elif snap_palette:
        # No quant step: snap every pixel to the nearest Teletext colour.
        arr = snap_to_palette(arr, 99999)

    if sharpen_amount > 0:
        from PIL.ImageFilter import UnsharpMask
        arr = np.array(
            Image.fromarray(arr).filter(
                UnsharpMask(radius=sharpen_radius, percent=sharpen_amount,
                            threshold=sharpen_threshold)),
            dtype=np.uint8)

    if snap > 0:
        arr = snap_to_palette(arr, snap)

    if dither:
        arr = floyd_steinberg_dither(arr)

    return Image.fromarray(arr)


def convert_image(img_path, use_hold=True, use_fill=True, use_sep=False, greedy=False, luma=False,
                  filter='bilinear', par=1.2,
                  sharpen_radius=0.0, sharpen_amount=0, sharpen_threshold=0,
                  gamma=1.0, contrast=1.0, saturation=1.0, linear=False,
                  dither=False, refine=False, quant_colors=0, posterize=0, median=0, snap=0,
                  snap_palette=False, smooth=0):
    """
    Load image, resize to fit 40x25 Mode 7 grid, encode each row.
    Returns a bytearray of 1000 bytes (MODE7_WIDTH * MODE7_HEIGHT).
    greedy: use fast greedy solver instead of the default full DP solver.
    par: pixel aspect ratio (sub-pixel width / height on display).
         1.0 = square pixels (emulator); 1.2 = LCD TV; 1.22 = CRT TV.
         Values > 1.0 pre-compress the image horizontally so the display stretches it back correctly.
    gamma: power-law tone adjustment applied before resize. >1 brightens, <1 darkens.
    contrast: PIL contrast enhancement factor applied before resize. >1 = more contrast.
    saturation: PIL colour saturation factor applied before resize. >1 = more saturated.
    sharpen_radius/amount/threshold: unsharp mask applied after resize, before DP.
         Pushes pixel values toward the extremes, improving Teletext palette matching.
         amount=0 disables sharpening.
    linear: linearise source pixels from sRGB to linear light before computing
         squared error. Corrects for gamma encoding: dark-tone differences are
         weighted more fairly. The Teletext palette (0/255 per channel) is unchanged.
    refine: run a local-search refinement pass after the solver.  At each position
         every valid candidate is tried; if substituting it (while re-simulating
         state forward through the fixed subsequent characters) reduces the total
         tail error the substitution is accepted.  Repeats until convergence.
    """
    preprocessed = preprocess_image(
        img_path, filter=filter, par=par,
        sharpen_radius=sharpen_radius, sharpen_amount=sharpen_amount,
        sharpen_threshold=sharpen_threshold,
        gamma=gamma, contrast=contrast, saturation=saturation, dither=dither,
        quant_colors=quant_colors, posterize=posterize, median=median, snap=snap,
        snap_palette=snap_palette)
    arr = np.array(preprocessed, dtype=np.uint8)
    pw, ph = preprocessed.size

    frame_w = pw // 2
    frame_h = ph // 3

    mode_str = ("greedy+refine" if greedy else "DP") + ("+refine" if (refine and not greedy) else "")
    print(f"Image resized to {pw}x{ph} px -> {frame_w}x{frame_h} chars  [{mode_str}]",
          file=sys.stderr)

    page = bytearray(MODE7_MAX_SIZE)
    for i in range(MODE7_MAX_SIZE):
        page[i] = MODE7_BLANK

    solver = greedy_row if greedy else dp_row

    def _solve_row(y7):
        et, gt = build_error_table(arr, y7, frame_w, luma=luma, linear=linear)
        row = solver(et, gt, frame_w,
                     use_hold=use_hold, use_fill=use_fill, use_sep=use_sep)
        if refine or greedy:   # refine is always applied after greedy; explicit --refine adds it to DP
            row = refine_row(row, et, gt, frame_w,
                             use_hold=use_hold, use_fill=use_fill, use_sep=use_sep)
        return row

    # Centre the image in the 40×25 grid.
    # Horizontal: colour-code col + frame_w image cols; centre the block.
    # Vertical: offset rows so the image sits in the middle of the 25 rows.
    col_off = (MODE7_WIDTH - (frame_w + 1)) // 2   # left edge of colour-code col
    row_off = (MODE7_HEIGHT - frame_h) // 2

    num_workers = min(frame_h, os.cpu_count() or 1)
    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for y7, row in enumerate(pool.map(_solve_row, range(frame_h))):
            completed += 1
            print(f"\rProcessing row {completed}/{frame_h}...", end='', file=sys.stderr)
            if smooth >= 2:
                row = _smooth_colour_runs(row, arr, y7, frame_w, smooth,
                                         use_hold=use_hold, use_fill=use_fill,
                                         use_sep=use_sep)
            base = (y7 + row_off) * MODE7_WIDTH
            page[base + col_off] = row[0]                          # colour code
            for xi in range(frame_w):
                page[base + col_off + 1 + xi] = row[FRAME_FIRST_COLUMN + xi]
            reset_col = col_off + 1 + frame_w
            if reset_col < MODE7_WIDTH:
                page[base + reset_col] = MODE7_BLACK_BG

    print(f"\rDone.                         ", file=sys.stderr)
    return page

# ---------------------------------------------------------------------------
# Teletext editor URL encoding (edit.tf / zxnet)
# ---------------------------------------------------------------------------

def _pack_page_b64(page):
    """Pack 1000 × 7-bit bytes into a base64 string (shared by all URL encoders)."""
    data = bytes(page)
    packed = bytearray()
    for i in range(0, len(data), 8):
        chunk = data[i:i+8]
        val = 0
        for b in chunk:
            val = (val << 7) | (b & 0x7F)
        for shift in range(48, -1, -8):
            packed.append((val >> shift) & 0xFF)
    return base64.urlsafe_b64encode(packed).decode('ascii').rstrip('=')

def to_edittf_url(page):
    """Return an edit.tf URL for the page: http://edit.tf/#0:<base64>"""
    return f'http://edit.tf/#0:{_pack_page_b64(page)}'

def to_zxnet_url(page):
    """Return a ZXNet teletext editor URL: https://zxnet.co.uk/teletext/editor/#0:<base64>"""
    return f'https://zxnet.co.uk/teletext/editor/#0:{_pack_page_b64(page)}'

# ---------------------------------------------------------------------------
# Preview PNG rendering
# ---------------------------------------------------------------------------

# Pixels per Mode 7 character cell in the preview image
PREVIEW_CELL_W = 16   # 2 sixels × 8 px
PREVIEW_CELL_H = 20   # 3 sixels × (20/3 ≈ 6.67 px) → use 20 total, split 7/7/6

# Sub-pixel row heights within a cell (sum = PREVIEW_CELL_H)
SIXEL_ROW_H = [7, 7, 6]
# Sub-pixel column widths within a cell (sum = PREVIEW_CELL_W)
SIXEL_COL_W = [8, 8]


def render_preview(page):
    """Render the Mode 7 page to a PIL Image."""
    out_w = MODE7_WIDTH  * PREVIEW_CELL_W
    out_h = MODE7_HEIGHT * PREVIEW_CELL_H
    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for row in range(MODE7_HEIGHT):
        # State at start of each row resets
        fg       = 7        # default white (overridden by col-0 colour code)
        bg       = 0        # black
        hold     = False
        last_gfx = MODE7_BLANK
        sep      = False

        y_px = row * PREVIEW_CELL_H

        for col in range(MODE7_WIDTH):
            byte = page[row * MODE7_WIDTH + col]
            x_px = col * PREVIEW_CELL_W

            # Apply Set-At effects before rendering (take effect at this cell)
            if byte == MODE7_NEW_BG:
                bg = fg
            elif byte == MODE7_BLACK_BG:
                bg = 0
            elif byte == MODE7_HOLD_GFX:
                hold = True
            elif byte == MODE7_RELEASE_GFX:
                hold = False
                last_gfx = MODE7_BLANK

            # Determine what's displayed at this cell
            if byte >= 128:
                display_char = last_gfx if hold else MODE7_BLANK
            else:
                display_char = byte

            # Render the 2×3 sixel grid
            y_off = 0
            for row_idx, rh in enumerate(SIXEL_ROW_H):
                x_off = 0
                for col_idx, cw in enumerate(SIXEL_COL_W):
                    sixel_idx = row_idx * 2 + col_idx
                    bit = GFX_PIXEL_BITS[sixel_idx]
                    is_set = bool(display_char & bit)
                    if sep and is_set:
                        # Separated graphics — each sub-pixel block has a gap on the
                        # left and a gap at the bottom; active fg area is top-right.
                        # (x*2+1)//3 rounds 2x/3 to the nearest integer.
                        act_w = (cw * 2 + 1) // 3
                        act_h = (rh * 2 + 1) // 3
                        gap_w = cw - act_w
                        img[y_px + y_off          : y_px + y_off + rh,
                            x_px + x_off          : x_px + x_off + cw] = COLOR_RGB[bg]
                        img[y_px + y_off          : y_px + y_off + act_h,
                            x_px + x_off + gap_w  : x_px + x_off + cw] = COLOR_RGB[fg]
                    else:
                        r, g, b = screen_rgb(is_set, fg, bg, sep=False)
                        img[y_px + y_off : y_px + y_off + rh,
                            x_px + x_off : x_px + x_off + cw] = (r, g, b)
                    x_off += cw
                y_off += rh

            # Apply Set-After effects (take effect from next cell)
            if byte >= 128:
                if MODE7_GFX_COLOUR < byte < MODE7_GFX_COLOUR + 8:
                    fg = byte - MODE7_GFX_COLOUR
                elif byte == MODE7_SEP_GFX:
                    sep = True
                elif byte == MODE7_CONTIG_GFX:
                    sep = False
            else:
                last_gfx = byte

    return Image.fromarray(img, 'RGB')

# ---------------------------------------------------------------------------
# BBC Micro DFS / SSD disk image writer
# ---------------------------------------------------------------------------

# Standard Acorn DFS / SSD constants
_SSD_TRACKS          = 80
_SSD_SECTORS_PER_TRK = 10
_SSD_SECTOR_SIZE     = 256
_SSD_TOTAL_SECTORS   = _SSD_TRACKS * _SSD_SECTORS_PER_TRK   # 800
_SSD_DISK_SIZE       = _SSD_TOTAL_SECTORS * _SSD_SECTOR_SIZE # 204 800
_SSD_MAX_FILES       = 31
_SSD_CATALOG_SECTORS = 2   # sectors 0+1 are the catalog

# Standard Acorn DFS catalog layout (sector 1):
#   byte 4: (write_cycle:4)(boot_opt:2)(total_sectors[9:8]:2)
#   byte 5: num_files * 8
#   byte 6: total_sectors[7:0]
_S1_CYCLE_BOOT_HIGH = 256 + 4   # disk[260]
_S1_NUM_FILES_X8    = 256 + 5   # disk[261]
_S1_SECTORS_LOW     = 256 + 6   # disk[262]


def _blank_ssd():
    """Return a fresh 80-track blank SSD image as a bytearray."""
    disk = bytearray(_SSD_DISK_SIZE)
    disk[0:8] = b'Mode7   '   # disk title (8 chars in sector 0)
    # sector 1 header
    high_bits = (_SSD_TOTAL_SECTORS >> 8) & 3   # = 3 for 800 sectors
    disk[_S1_CYCLE_BOOT_HIGH] = high_bits        # cycle=0, boot=none, sector high bits
    disk[_S1_NUM_FILES_X8]    = 0                # no files yet
    disk[_S1_SECTORS_LOW]     = _SSD_TOTAL_SECTORS & 0xFF  # = 0x20
    return disk


def write_to_ssd(ssd_path, bbc_name, data, load_addr=0x7C00, exec_addr=0x7C00):
    """
    Add or overwrite a file in a BBC Micro DFS .ssd disk image.

    ssd_path  : path to the .ssd file (created as a blank 80-track disk if absent)
    bbc_name  : DFS filename, max 7 alphanumeric characters (stored in $ directory)
    data      : bytes/bytearray of file content
    load_addr : 18-bit load address (default 0x7C00 = Mode 7 screen)
    exec_addr : 18-bit exec address (default 0x7C00)
    """
    ssd_path = Path(ssd_path)

    # Load existing disk or create blank
    if ssd_path.exists():
        disk = bytearray(ssd_path.read_bytes())
        if len(disk) < _SSD_DISK_SIZE:
            disk.extend(bytes(_SSD_DISK_SIZE - len(disk)))
    else:
        disk = _blank_ssd()

    # Sanitise BBC filename: uppercase alphanumeric, max 7 chars
    bbc_name = ''.join(c for c in bbc_name.upper() if c.isalnum())[:7] or 'MODE7'
    bbc_padded = bbc_name.ljust(7)[:7]

    num_files = disk[_S1_NUM_FILES_X8] // 8

    # Check whether this filename already exists in the $ directory
    existing_idx = -1
    for i in range(num_files):
        s0 = 8 + i * 8
        ename = disk[s0:s0 + 7].decode('ascii', errors='replace').rstrip()
        edir  = chr(disk[s0 + 7] & 0x7F)
        if ename.upper() == bbc_name.upper() and edir == '$':
            existing_idx = i
            break

    file_len      = len(data)
    secs_needed   = (file_len + _SSD_SECTOR_SIZE - 1) // _SSD_SECTOR_SIZE

    if existing_idx >= 0:
        # Overwrite: reuse the same start sector
        s1 = 256 + 8 + existing_idx * 8
        start_sector = disk[s1 + 7] | (((disk[s1 + 6] >> 1) & 1) << 8)
    else:
        # New file: place it after all existing data
        if num_files >= _SSD_MAX_FILES:
            raise ValueError(f'Disk catalog full ({_SSD_MAX_FILES} files maximum)')
        start_sector = _SSD_CATALOG_SECTORS
        for i in range(num_files):
            s1 = 256 + 8 + i * 8
            flen = (disk[s1 + 4]
                    | (disk[s1 + 5] << 8)
                    | (((disk[s1 + 6] >> 2) & 3) << 16))
            fss  = disk[s1 + 7] | (((disk[s1 + 6] >> 1) & 1) << 8)
            end  = fss + (flen + _SSD_SECTOR_SIZE - 1) // _SSD_SECTOR_SIZE
            if end > start_sector:
                start_sector = end
        free = _SSD_TOTAL_SECTORS - start_sector
        if secs_needed > free:
            raise ValueError(
                f'Disk full: need {secs_needed} sectors, only {free} available')

    # Write file data (zero-pad to a whole number of sectors)
    off = start_sector * _SSD_SECTOR_SIZE
    disk[off : off + file_len] = data
    disk[off + file_len : off + secs_needed * _SSD_SECTOR_SIZE] = bytes(
        secs_needed * _SSD_SECTOR_SIZE - file_len)

    # Write / update catalog entry
    entry_idx = existing_idx if existing_idx >= 0 else num_files

    s0 = 8 + entry_idx * 8
    disk[s0 : s0 + 7] = bbc_padded.encode('ascii')
    disk[s0 + 7] = ord('$')   # directory '$', not locked

    s1 = 256 + 8 + entry_idx * 8
    disk[s1 + 0] = load_addr & 0xFF
    disk[s1 + 1] = (load_addr >> 8) & 0xFF
    disk[s1 + 2] = exec_addr & 0xFF
    disk[s1 + 3] = (exec_addr >> 8) & 0xFF
    disk[s1 + 4] = file_len & 0xFF
    disk[s1 + 5] = (file_len >> 8) & 0xFF
    misc = (((exec_addr  >> 16) & 3) << 6 |
            ((load_addr  >> 16) & 3) << 4 |
            ((file_len   >> 16) & 3) << 2 |
            ((start_sector >> 8) & 1) << 1)
    disk[s1 + 6] = misc
    disk[s1 + 7] = start_sector & 0xFF

    # Update file count if adding new file
    if existing_idx < 0:
        disk[_S1_NUM_FILES_X8] = (num_files + 1) * 8

    # Bump write-cycle counter (bits 7-4 of catalog header byte)
    cycle = ((disk[_S1_CYCLE_BOOT_HIGH] >> 4) + 1) & 0xF
    disk[_S1_CYCLE_BOOT_HIGH] = (disk[_S1_CYCLE_BOOT_HIGH] & 0x0F) | (cycle << 4)

    ssd_path.write_bytes(disk)

    verb = 'Updated' if existing_idx >= 0 else 'Added'
    print(f'{verb} $.{bbc_name} on {ssd_path}  '
          f'(sector {start_sector}, {file_len} bytes, '
          f'load/exec=&{load_addr:X})')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Presets — named combinations of options for common use cases.
# Individual flags always override the preset value.
# ---------------------------------------------------------------------------

PRESETS = {
    'photo': dict(
        # Photographic sources: portraits, landscapes, general images.
        # Boosts saturation to exploit the fully-saturated Teletext palette,
        # lifts contrast slightly, and applies moderate sharpening.
        saturation=1.8, contrast=1.3,
        sharpen_amount=150, sharpen_radius=1.0, sharpen_threshold=3,
    ),
    'clean': dict(
        # A smoother alternative to 'photo' when the result looks noisy or speckled.
        # Light median denoising before resize + moderate colour snap to pull
        # near-palette pixels in, reducing ambiguous mid-tones without losing edges.
        median=1, snap=25,
        saturation=1.8, contrast=1.3,
        sharpen_amount=120, sharpen_radius=1.0, sharpen_threshold=5,
    ),
    'smooth': dict(
        # Noise reduction emphasis: best for noisy JPEGs, soft gradients, portraits.
        # 5×5 median removes small blobs before resize; palette pre-quantisation
        # limits colour complexity; snap eliminates residual mid-tones.
        median=2, snap=35, quant_colors=16,
        saturation=1.8, contrast=1.2,
        sharpen_amount=80, sharpen_radius=1.5, sharpen_threshold=5,
    ),
    'vivid': dict(
        # High-impact look with punchy colours and strong edge definition.
        # Good for images that need to read clearly at a distance.
        saturation=2.5, contrast=1.5,
        sharpen_amount=200, sharpen_radius=1.0,
    ),
    'graphic': dict(
        # Flat-colour artwork, logos, cartoons, pixel art.
        # Strong sharpening with a tight radius to preserve hard edges;
        # high saturation to snap to palette colours; no threshold so every
        # edge is enhanced.
        saturation=2.0, contrast=1.5,
        sharpen_amount=300, sharpen_radius=0.5, sharpen_threshold=0,
    ),
    'flat': dict(
        # Bold, graphic style with deliberately limited colours.
        # Posterise reduces tonal steps to hard bands; aggressive snap forces
        # remaining mid-tones to the nearest Teletext colour; high saturation
        # and contrast push everything toward the palette extremes.
        # Good for logos, illustration-style images, or a punchy retro look.
        posterize=3, snap=70,
        saturation=2.5, contrast=1.8,
        sharpen_amount=200, sharpen_radius=0.5, sharpen_threshold=0,
    ),
    'retro': dict(
        # Mimics the limited, blocky look of real Ceefax pages.
        # Pre-quantises to 8 colours and snaps aggressively to the Teletext
        # palette, producing flat regions with minimal dithering noise.
        # Less detail but very authentic.
        quant_colors=8, snap=80,
        saturation=2.2, contrast=1.4,
        sharpen_amount=100, sharpen_radius=1.0,
        par=1.2,
    ),
    'art': dict(
        # Teletext-art-style output: quantise to 6 dominant colours then snap
        # each region unconditionally to the nearest Teletext colour.
        # Produces smooth colour boundaries rather than per-pixel noise —
        # the hallmark of hand-crafted teletext art.
        # Pairs well with --smooth 3 for further run cleanup.
        quant_colors=6, snap_palette=True,
        saturation=2.0, contrast=1.4,
        sharpen_amount=120, sharpen_radius=1.0, sharpen_threshold=3,
    ),
    'dark': dict(
        # Dark or underexposed source images.
        # Gamma lift brightens shadows; modest contrast and saturation boost.
        gamma=1.8, saturation=1.5, contrast=1.4,
        sharpen_amount=100, sharpen_radius=1.0, sharpen_threshold=5,
    ),
    'tv': dict(
        # Viewing on a modern LCD television.
        # Applies the correct PAR for Ceefax-era display (1.2) and the photo
        # colour/sharpening settings.
        par=1.2,
        saturation=1.8, contrast=1.3,
        sharpen_amount=150, sharpen_radius=1.0, sharpen_threshold=3,
    ),
    'crt': dict(
        # Viewing on a CRT television.
        # Uses the mathematically derived PAR for the SAA5050 on a PAL CRT.
        par=1.22,
        saturation=1.8, contrast=1.3,
        sharpen_amount=150, sharpen_radius=1.0, sharpen_threshold=3,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description='Convert a PNG image to Teletext/Mode 7 graphics (40×25 chars, 1000 bytes).')
    parser.add_argument('input', help='Input PNG image')
    parser.add_argument('-o', '--output', help='Output .bin file (default: <input>.bin)')
    parser.add_argument('--preview', metavar='PNG', help='Save a preview PNG of the rendered Teletext output')
    parser.add_argument('--url', action='store_true', help='Print an edit.tf URL for the output')
    parser.add_argument('--zxnet', action='store_true', help='Print a ZXNet teletext editor URL for the output')
    parser.add_argument('--nohold', action='store_true', help='Disable Hold Graphics optimisation')
    parser.add_argument('--nofill', action='store_true', help='Disable New Background optimisation')
    parser.add_argument('--sep', action='store_true', help='Enable Separated Graphics mode (experimental)')
    parser.add_argument('--greedy', action='store_true',
                        help='Use fast greedy solver instead of the default full DP solver. '
                             'Automatically applies the local-search refinement pass, giving '
                             'good quality at ~4× the speed of DP. Useful for quick previews '
                             'or when DP is too slow.')
    parser.add_argument('--refine', action='store_true',
                        help='Run a local-search refinement pass after the solver. '
                             'At each character position every valid candidate is tried; '
                             'if substituting it while re-simulating the decoder state '
                             'through the fixed subsequent characters reduces the total '
                             'tail error the substitution is accepted. '
                             'Passes repeat until convergence (typically 1-3 passes). '
                             'Works after both --greedy and the default DP solver. '
                             'Adds modest extra time per row (O(frame_w² × candidates)).')
    parser.add_argument('--luma', action='store_true',
                        help='Use perceptual luminance weighting (ITU-R BT.601) for error metric')
    parser.add_argument('--dither', action='store_true',
                        help='Apply Floyd-Steinberg error diffusion dithering at sub-pixel level '
                             'after resize and sharpening. Quantizes each sub-pixel to the nearest '
                             'Teletext palette colour and diffuses the error to neighbours '
                             '(7/16 right, 5/16 below, 3/16 below-left, 1/16 below-right). '
                             'May improve gradients and tone range but can create visible noise '
                             'patterns at Mode 7\'s low resolution. Best combined with '
                             '--saturation to push source colours toward the palette first.')
    parser.add_argument('--linear', action='store_true',
                        help='Linearise source pixels from sRGB to linear light before computing '
                             'squared error. Source images are gamma-encoded (~2.2); computing '
                             'error in gamma space underweights dark-tone differences. '
                             'The Teletext palette (all 0 or 255 per channel) is unchanged. '
                             'Can be combined with --luma for both corrections simultaneously.')
    parser.add_argument('--filter', choices=['bilinear', 'lanczos', 'bicubic', 'nearest', 'cimg'],
                        default='bilinear',
                        help='Resampling filter for image resize (default: bilinear). '
                             'cimg matches the left-aligned nearest-neighbour used by the C++ executable.')
    parser.add_argument('--par', type=float, default=1.2,
                        metavar='RATIO',
                        help='Pixel aspect ratio of the target display (sub-pixel width / height). '
                             '1.2 = modern LCD TV / recommended (default; matches Ceefax broadcast '
                             'footage; 480 teletext pixels * 1.2 = 576 PAL active lines). '
                             '1.0 = square pixels (suits emulators with 1:1 pixel mapping). '
                             '1.22 = CRT TV (mathematically derived: 768 / (52.6us * 6MHz * 2) = 1.22; '
                             'SAA5050 chip on a CRT with 768 square-pixel PAL width).')
    parser.add_argument('--sharpen-radius', type=float, default=1.0, metavar='R',
                        help='Unsharp mask blur radius in pixels, applied to the resized sub-pixel '
                             'image (default: 1.0). Controls the spatial extent of edge enhancement. '
                             '0.5 = sub-pixel edges only; 1.0 = one-cell edges (recommended starting '
                             'point); 2.0 = two-cell halos, good for enhancing broad colour regions. '
                             'Values above 3.0 rarely help at Mode 7 resolution.')
    parser.add_argument('--sharpen-amount', type=int, default=0, metavar='PCT',
                        help='Unsharp mask strength as a percentage (default: 0 = disabled). '
                             'Pushes pixel values toward channel extremes, reducing distance to the '
                             'nearest Teletext palette colour before the DP solver runs. '
                             '50-100 = subtle, good for photographic sources; '
                             '100-200 = moderate, recommended starting point for most images; '
                             '200-300 = strong, useful for graphics/logos with hard colour edges; '
                             'above 500 will posterise heavily (may be intentional).')
    parser.add_argument('--sharpen-threshold', type=int, default=0, metavar='T',
                        help='Unsharp mask threshold: minimum per-channel difference required before '
                             'sharpening is applied (default: 0 = sharpen everywhere, 0-255 range). '
                             '0 = sharpen all pixels including smooth gradients and noise; '
                             '3-10 = skip near-flat areas, avoids amplifying compression artefacts; '
                             '10-20 = only sharpen pronounced edges, best for noisy sources.')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                        help='Power-law tone adjustment applied before resize (default: 1.0 = off). '
                             'output = (input/255)^(1/G) * 255. '
                             '1.5-2.2 = brighten (lift shadows, good for dark or underexposed images); '
                             '2.2 = approximates linearising a typical sRGB-encoded source; '
                             '0.5-0.8 = darken (good for washed-out or high-key images).')
    parser.add_argument('--contrast', type=float, default=1.0, metavar='C',
                        help='Contrast enhancement factor applied before resize (default: 1.0 = off). '
                             '0.0 = flat grey; 1.0 = original; '
                             '1.2-1.5 = subtle boost, good for most images; '
                             '1.5-2.5 = strong boost, useful for low-contrast or foggy sources; '
                             'above 3.0 will clip highlights and shadows heavily.')
    parser.add_argument('--posterize', type=int, default=0, metavar='BITS',
                        help='Posterise to BITS bits per channel before resize (0 = off, 1–7). '
                             '1 bit = only 0/255 per channel (8 colours); '
                             '2 bits = 4 values per channel; 3–4 suits most photos.')
    parser.add_argument('--smooth', type=int, default=0, metavar='N',
                        help='Merge colour runs shorter than N cells into the dominant '
                             'neighbouring colour after solving (0 = off). '
                             'Eliminates salt-and-pepper colour noise from automated '
                             'conversion, where the solver switches colour for just 1–2 '
                             'cells. Re-renders affected cells with the new colour. '
                             'Try 2–3 for subtle cleanup; 4–6 for a bolder, more '
                             'hand-drawn look.')
    parser.add_argument('--snap', type=int, default=0, metavar='T',
                        help='Snap pixels within Euclidean RGB distance T of a Teletext palette '
                             'colour to that colour before dithering (0 = off, 1–255). '
                             'Reduces ambiguous mid-tones that cause noisy dithering patterns. '
                             'Try 20–40 for a subtle effect; 60–80 clips more aggressively. '
                             'Applied after resize, quantisation and sharpening.')
    parser.add_argument('--snap-palette', action='store_true', default=False,
                        help='After --quant, snap every quantized colour region unconditionally to '
                             'its nearest Teletext palette colour. Produces the "art palette" look: '
                             'smooth region boundaries (from quantization) where each region is a '
                             'pure Teletext colour. Without --quant, snaps all pixels regardless '
                             'of distance. Pairs well with --quant 4–8.')
    parser.add_argument('--median', type=int, default=0, metavar='RADIUS',
                        help='Apply a median filter of (2*RADIUS+1)×(2*RADIUS+1) pixels before resize '
                             '(0 = off). Removes noise and small colour blobs while preserving hard '
                             'edges — better than Gaussian blur for photos. '
                             '1 = 3×3 (gentle); 2 = 5×5; 3 = 7×7 (stronger). '
                             'Good for reducing JPEG artefacts and speckle in the output.')
    parser.add_argument('--quant', type=int, default=0, metavar='N',
                        help='Pre-quantise to N colours after resize (0 = off). '
                             'Reduces colour complexity, producing flatter regions '
                             'and a cleaner output. Try 8–32 for photos.')
    parser.add_argument('--saturation', type=float, default=1.0, metavar='S',
                        help='Colour saturation factor applied before resize (default: 1.0 = off). '
                             '0.0 = greyscale; 1.0 = original; '
                             '1.5-2.0 = recommended for photographic sources (Teletext palette is '
                             'fully saturated so boosting helps snap colours to the nearest entry); '
                             '3.0+ = vivid/posterised look.')
    parser.add_argument('--ssd', metavar='DISK.SSD',
                        help='Add output to a BBC Micro DFS .ssd disk image '
                             '(80-track, created if it does not exist)')
    parser.add_argument('--ssd-name', metavar='NAME',
                        help='BBC DFS filename on the disk (max 7 chars, '
                             'default: input filename stem)')
    parser.add_argument('--preset', choices=sorted(PRESETS),
                        help='Named combination of options for common use cases. '
                             'Any explicit flag overrides the preset value. '
                             'photo: balanced portraits/landscapes; '
                             'clean: like photo with light denoising and snap (less speckle); '
                             'smooth: heavy noise reduction, best for noisy JPEGs; '
                             'vivid: punchy colours, strong edges; '
                             'graphic: logos/cartoons (tight sharpening, high saturation); '
                             'flat: bold posterised look with limited palette; '
                             'retro: authentic Ceefax style, blocky limited colours; '
                             'dark: underexposed images (gamma lift); '
                             'tv: LCD TV viewing (PAR 1.2 + photo settings); '
                             'crt: CRT TV viewing (PAR 1.22 + photo settings).')

    # Two-pass parse: first pass reads --preset, then its values become
    # defaults so any explicitly-provided flags still take precedence.
    args, _ = parser.parse_known_args()
    if args.preset:
        parser.set_defaults(**PRESETS[args.preset])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'Error: {input_path} not found', file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix('.bin')

    page = convert_image(
        input_path,
        use_hold=not args.nohold,
        use_fill=not args.nofill,
        greedy=args.greedy,
        use_sep=args.sep,
        luma=args.luma,
        filter=args.filter,
        par=args.par,
        sharpen_radius=args.sharpen_radius,
        sharpen_amount=args.sharpen_amount,
        sharpen_threshold=args.sharpen_threshold,
        gamma=args.gamma,
        contrast=args.contrast,
        saturation=args.saturation,
        linear=args.linear,
        dither=args.dither,
        refine=args.refine,
        quant_colors=args.quant,
        posterize=args.posterize,
        median=args.median,
        snap=args.snap,
        snap_palette=args.snap_palette,
        smooth=args.smooth,
    )

    # Write binary
    output_path.write_bytes(page)
    print(f'Written {len(page)} bytes -> {output_path}')

    # Preview
    if args.preview:
        preview_img = render_preview(page)
        preview_img.save(args.preview)
        print(f'Preview saved -> {args.preview}')

    # URLs
    if args.url:
        print(to_edittf_url(page))
    if args.zxnet:
        print(to_zxnet_url(page))

    # SSD disk image
    if args.ssd:
        dfs_name = args.ssd_name if args.ssd_name else input_path.stem
        write_to_ssd(args.ssd, dfs_name, bytes(page))


if __name__ == '__main__':
    # Raise Python recursion limit for the DP solver (max depth = 40)
    sys.setrecursionlimit(10000)
    main()
