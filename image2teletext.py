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

# ---------------------------------------------------------------------------
# Precompute per-row error lookup table (numpy vectorised)
# ---------------------------------------------------------------------------

def build_error_table(img, y7, frame_w, luma=False):
    """
    Return err_table[x7_idx, fg, bg, sep, screen_char] = total squared RGB error
    for displaying screen_char (0-127) at character column x7_idx in row y7,
    with the given fg/bg palette indices and sep flag.

    Also returns gfx_table[x7_idx, fg, bg, sep] = optimal graphics character.
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


def _best_initial_colour(err_table, gfx_table):
    """Choose the starting fg colour (1-7) that gives lowest error on column 0."""
    min_err = 10 ** 18
    min_col = 7
    for fg in range(7, 0, -1):
        gc  = int(gfx_table[0, fg, 0, 0])
        err = int(err_table[0, fg, 0, 0, gc])
        if err < min_err:
            min_err = err
            min_col = fg
    return min_col


# ---------------------------------------------------------------------------
# Fast greedy solver (default — 2-step lookahead)
# ---------------------------------------------------------------------------

def greedy_row(err_table, gfx_table, frame_w,
               use_hold=True, use_fill=True, use_sep=False):
    """
    Greedy row encoder with 2-step lookahead.  Fast (~ms per row).
    err_table, gfx_table: precomputed from build_error_table().
    """
    row_end = FRAME_FIRST_COLUMN + frame_w
    min_col = _best_initial_colour(err_table, gfx_table)

    result = [MODE7_BLANK] * MODE7_WIDTH
    result[0] = MODE7_GFX_COLOUR + min_col
    state = pack_state(min_col, 0, False, MODE7_BLANK, False)

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
                xi2 = xi + 1
                ngc = int(gfx_table[xi2, nfg, nbg, nsep])
                ndc, neff_bg = _effective_cell(ngc, nfg, nbg, nhold, nlast, nsep)
                err += int(err_table[xi2, nfg, neff_bg, nsep, ndc])

            if err < best_err:
                best_err  = err
                best_char = char

        result[x7] = best_char
        state = next_state(best_char, state, use_hold, use_fill, use_sep)

    return result


# ---------------------------------------------------------------------------
# Full DP solver (--slow — near-optimal, but slow for complex images)
# ---------------------------------------------------------------------------

def dp_row(err_table, gfx_table, frame_w,
           use_hold=True, use_fill=True, use_sep=False):
    """
    Vectorised bottom-up DP row encoder.  Matches C++ -slow quality:
    tries all 63 non-blank graphics characters per state so the DP can
    account for HOLD_GFX picking up a non-locally-optimal gfx char later.
    """
    all_s  = np.arange(MAX_STATE, dtype=np.int32)
    fg_v   = all_s & 7
    bg_v   = (all_s >> 3) & 7
    hold_v = (all_s >> 6) & 1
    last_v = (all_s >> 7) & 0x7F
    sep_v  = (all_s >> 14) & 1
    bg_zero = np.zeros(MAX_STATE, dtype=np.int32)
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
        hl_release = np.full(MAX_STATE, np.int32(MODE7_BLANK << 7), dtype=np.int32)
    else:
        _hl = np.full(MAX_STATE, np.int32(MODE7_BLANK << 7), dtype=np.int32)
        hl_blank = hl_ctrl = hl_hold = hl_release = _hl

    # All 63 non-blank graphics characters (bit5 maps to bit6 for BR pixel)
    GFX_CHARS = np.array([MODE7_BLANK | (i & 0x1f) | ((i & 0x20) << 1)
                          for i in range(1, 64)], dtype=np.int32)  # (63,)
    NUM_GFX = len(GFX_CHARS)

    # Next-state for each gfx char × each state (precomputed before column loop)
    base_ns = (sep_cur | (hold_v << 6) | (bg_v << 3) | fg_v).astype(np.int32)  # (MAX_STATE,)
    if use_hold:
        ns_gfx = base_ns[None, :] | (GFX_CHARS[:, None] << 7)  # (63, MAX_STATE)
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
    ns_stack   = np.empty((NC + 1, MAX_STATE), dtype=np.int32)  # +1 for gfx slot
    char_stack = np.empty((NC + 1, MAX_STATE), dtype=np.int32)
    mask_stack = np.ones ((NC + 1, MAX_STATE), dtype=bool)

    for i, (ch, ns, msk, _) in enumerate(ctrl_cands):
        ns_stack[i]   = ns
        char_stack[i] = ch
        if msk is not None:
            mask_stack[i] = msk
    # Row NC is the gfx slot: char updated per column, ns not used (cost pre-computed)

    # Precomputed index arrays for fast per-column lookups
    # err_table[xi] shape (8,8,2,128) → C-order flat first-3 dim: fg*16 + bg*2 + sep
    fg_bg_sep_idx      = (fg_v * 16 + bg_v * 2 + sep_v).astype(np.int32)  # (MAX_STATE,)
    fg_bg_sep_fg_idx   = (fg_v * 16 + fg_v * 2 + sep_v).astype(np.int32)  # bg=fg
    fg_bg_sep_zero_idx = (fg_v * 16 +           sep_v ).astype(np.int32)   # bg=0
    # dp_next reshaped as (2, 128, 128) for gfx future-cost lookups
    # Combined (sep, base_lower) index: 2*128=256 unique values
    base_lower_v    = (all_s & np.int32(0x7F)).astype(np.int32)            # (MAX_STATE,)
    sep_base_low_v  = (sep_v * np.int32(128) + base_lower_v).astype(np.int32)  # 0..255

    # ------------------------------------------------------------------
    # Preallocate per-column workspace
    # ------------------------------------------------------------------
    dp_next       = np.zeros(MAX_STATE, dtype=np.int32)
    best_char_arr = np.full((frame_w, MAX_STATE), MODE7_BLANK, dtype=np.int32)
    err_mat       = np.empty((NC,     MAX_STATE), dtype=np.int32)
    total_mat     = np.empty((NC + 1, MAX_STATE), dtype=np.int32)
    gfx_work      = np.empty((MAX_STATE, NUM_GFX), dtype=np.int32)  # (32768, 63)

    for xi in range(frame_w - 1, -1, -1):

        et_flat = err_table[xi].reshape(128, 128)       # (fg_bg_sep, char) view, free

        # ----------------------------------------------------------
        # Gfx candidates — two-level to stay cache-friendly
        # Level 1: find best gfx char per state via (MAX_STATE, 63) argmin axis=1
        #   source tables: (128, 63) err compact and (2,128,128) dp_next → fit in cache
        # ----------------------------------------------------------
        # Error for each gfx char per state: gather from (128, 63) compact table
        gfx_err_compact = et_flat[:, GFX_CHARS]         # (128, 63) — 32KB, L1 friendly
        np.take(gfx_err_compact, fg_bg_sep_idx,
                axis=0, out=gfx_work)                   # (32768, 63) gather from 32KB

        # dp_next future-cost for gfx next-states.
        # Build a compact (256, 63) table indexed by (sep*128+base_lower, gfx_char_slot).
        # Source dp_next_3d is (2,128,128)=128KB; slice to (256,63)=64KB → fits in L2.
        if use_hold:
            # dp_next_3d[:, GFX_CHARS, :] shape (2,63,128) → reshape+transpose → (256,63)
            dp_gfx_fut = dp_next.reshape(2, 128, 128)[:, GFX_CHARS, :].transpose(0,2,1).reshape(256, NUM_GFX)
            gfx_work += dp_gfx_fut[sep_base_low_v]   # gather from (256,63) → (32768,63)
        else:
            fut_1d = dp_next.reshape(2, 128, 128)[sep_v, MODE7_BLANK, base_lower_v]
            gfx_work += fut_1d[:, None]               # broadcast (32768,1)

        # argmin axis=1: cache-friendly (63 contiguous values per state)
        best_gfx_j    = gfx_work.argmin(axis=1)         # (32768,)
        best_gfx_cost = gfx_work[all_s, best_gfx_j]     # (32768,)
        char_stack[NC] = GFX_CHARS[best_gfx_j]          # update gfx char slot

        # ----------------------------------------------------------
        # Control candidates — (NC, 32768) as before
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

        best_c = np.argmin(total_mat, axis=0)            # (MAX_STATE,) over NC+1 rows
        best_char_arr[xi] = char_stack[best_c, all_s]
        dp_next = total_mat[best_c, all_s]

    # Forward simulation from the initial state to recover the optimal sequence
    min_col    = _best_initial_colour(err_table, gfx_table)
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

def convert_image(img_path, use_hold=True, use_fill=True, use_sep=False, slow=False, luma=False,
                  filter='bilinear'):
    """
    Load image, resize to fit 40x25 Mode 7 grid, encode each row.
    Returns a bytearray of 1000 bytes (MODE7_WIDTH * MODE7_HEIGHT).
    slow: use full DP solver (near-optimal quality, slower).
    """
    img = Image.open(img_path).convert('RGB')
    iw, ih = img.size

    # Maintain aspect ratio, fit within 78x75 pixel canvas
    pw = MODE7_PIXEL_W
    ph = pw * ih // iw
    if ph % 3:
        ph += 3 - (ph % 3)
    if ph > MODE7_PIXEL_H:
        ph = MODE7_PIXEL_H
        pw = ph * iw // ih
        if pw % 2:
            pw += 1

    img = img.resize((pw, ph), _FILTER_MAP[filter])
    arr = np.array(img, dtype=np.uint8)

    frame_w = pw // 2
    frame_h = ph // 3

    mode_str = "DP (slow)" if slow else "greedy"
    print(f"Image resized to {pw}x{ph} px -> {frame_w}x{frame_h} chars  [{mode_str}]",
          file=sys.stderr)

    page = bytearray(MODE7_MAX_SIZE)
    for i in range(MODE7_MAX_SIZE):
        page[i] = MODE7_BLANK

    solver = dp_row if slow else greedy_row

    def _solve_row(y7):
        et, gt = build_error_table(arr, y7, frame_w, luma=luma)
        return solver(et, gt, frame_w,
                      use_hold=use_hold, use_fill=use_fill, use_sep=use_sep)

    num_workers = min(frame_h, os.cpu_count() or 1)
    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for y7, row in enumerate(pool.map(_solve_row, range(frame_h))):
            completed += 1
            print(f"\rProcessing row {completed}/{frame_h}...", end='', file=sys.stderr)
            for x7 in range(MODE7_WIDTH):
                page[y7 * MODE7_WIDTH + x7] = row[x7]
            if FRAME_FIRST_COLUMN + frame_w < MODE7_WIDTH:
                page[y7 * MODE7_WIDTH + FRAME_FIRST_COLUMN + frame_w] = MODE7_BLACK_BG

    print(f"\rDone.                         ", file=sys.stderr)
    return page

# ---------------------------------------------------------------------------
# edit.tf URL encoding
# ---------------------------------------------------------------------------

def to_edittf_url(page):
    """
    Pack 1000 bytes (7 bits each) into 875 bytes, then base64-encode,
    producing a URL for http://edit.tf/#0:<base64>
    """
    data = bytes(page)
    n = len(data)  # 1000

    # Pack 8×7-bit values into 7 bytes
    packed = bytearray()
    for i in range(0, n, 8):
        chunk = data[i:i+8]
        # Build a 56-bit integer from 8 × 7-bit values
        val = 0
        for b in chunk:
            val = (val << 7) | (b & 0x7F)
        # Extract 7 bytes (8 bits each)
        for shift in range(48, -1, -8):
            packed.append((val >> shift) & 0xFF)

    b64 = base64.urlsafe_b64encode(packed).decode('ascii').rstrip('=')
    return f'http://edit.tf/#0:{b64}'

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
                    r, g, b = screen_rgb(is_set, fg, bg, sep)
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

def main():
    parser = argparse.ArgumentParser(
        description='Convert a PNG image to Teletext/Mode 7 graphics (40×25 chars, 1000 bytes).')
    parser.add_argument('input', help='Input PNG image')
    parser.add_argument('-o', '--output', help='Output .bin file (default: <input>.bin)')
    parser.add_argument('--preview', metavar='PNG', help='Save a preview PNG of the rendered Teletext output')
    parser.add_argument('--url', action='store_true', help='Print an edit.tf URL for the output')
    parser.add_argument('--nohold', action='store_true', help='Disable Hold Graphics optimisation')
    parser.add_argument('--nofill', action='store_true', help='Disable New Background optimisation')
    parser.add_argument('--sep', action='store_true', help='Enable Separated Graphics mode (experimental)')
    parser.add_argument('--slow', action='store_true',
                        help='Use full DP solver (near-optimal quality, much slower)')
    parser.add_argument('--luma', action='store_true',
                        help='Use perceptual luminance weighting (ITU-R BT.601) for error metric')
    parser.add_argument('--filter', choices=['bilinear', 'lanczos', 'bicubic', 'nearest'],
                        default='bilinear',
                        help='Resampling filter for image resize (default: bilinear)')
    parser.add_argument('--ssd', metavar='DISK.SSD',
                        help='Add output to a BBC Micro DFS .ssd disk image '
                             '(80-track, created if it does not exist)')
    parser.add_argument('--ssd-name', metavar='NAME',
                        help='BBC DFS filename on the disk (max 7 chars, '
                             'default: input filename stem)')
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
        slow=args.slow,
        use_sep=args.sep,
        luma=args.luma,
        filter=args.filter,
    )

    # Write binary
    output_path.write_bytes(page)
    print(f'Written {len(page)} bytes -> {output_path}')

    # Preview
    if args.preview:
        preview_img = render_preview(page)
        preview_img.save(args.preview)
        print(f'Preview saved -> {args.preview}')

    # URL
    if args.url:
        url = to_edittf_url(page)
        print(url)

    # SSD disk image
    if args.ssd:
        dfs_name = args.ssd_name if args.ssd_name else input_path.stem
        write_to_ssd(args.ssd, dfs_name, bytes(page))


if __name__ == '__main__':
    # Raise Python recursion limit for the DP solver (max depth = 40)
    sys.setrecursionlimit(10000)
    main()
