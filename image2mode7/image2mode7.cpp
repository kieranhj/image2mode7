// image2mode7.cpp : Defines the entry point for the console application.
//
// Generic image -> mode 7 (aka Teletext) conversion routine
// Based on an initial algorithm by Puppeh (Julian Brown)
//

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>

#include "CImg.h"

extern "C"
{
#include "b64/cencode.h"
}

using namespace cimg_library;

#define MODE7_WIDTH			40
#define MODE7_HEIGHT		25
#define MODE7_MAX_SIZE		(MODE7_WIDTH * MODE7_HEIGHT)

#define IMAGE_W				(src._width)
#define IMAGE_H				(src._height)

#define MODE7_PIXEL_W		78
#define MODE7_PIXEL_H		75

#define FRAME_WIDTH			(frame_width)
#define FRAME_HEIGHT		(frame_height)
#define FRAME_SIZE			(MODE7_WIDTH * FRAME_HEIGHT)
#define FRAME_FIRST_COLUMN	1				// (MODE7_WIDTH - FRAME_WIDTH)

#define MODE7_BLANK			32
#define MODE7_BLACK_BG		156
#define MODE7_NEW_BG		157
#define MODE7_HOLD_GFX		158
#define MODE7_RELEASE_GFX	159
#define MODE7_GFX_COLOUR	144
#define MODE7_CONTIG_GFX	153
#define MODE7_SEP_GFX		154

#define CLAMP(a,low,high)	((a) < (low) ? (low) : ((a) > (high) ? (high) : (a)))
#define THRESHOLD(a,t)		((a) >= (t) ? 255 : 0)

#define MAX_STATE			(1U << 15)
#define GET_STATE(fg,bg,hold_mode,last_gfx_char,sep)	( (sep) << 14 | (last_gfx_char) << 7 | (hold_mode) << 6 | ((bg) << 3) | (fg))

#define IMAGE_X_FROM_X7(x7)	(((x7) - FRAME_FIRST_COLUMN) * 2)
#define IMAGE_Y_FROM_Y7(x7)	((y7) * 3)

#define MAX_3(A,B,C)		((A)>(B)?((A)>(C)?(A):(C)):(B)>(C)?(B):(C))
#define MIN_3(A,B,C)		((A)<(B)?((A)<(C)?(A):(C)):(B)<(C)?(B):(C))

#define _COLOUR_DEBUG		FALSE

static CImg<unsigned char> src;
static unsigned char mode7[MODE7_MAX_SIZE];

static int total_error_in_state[MAX_STATE][MODE7_WIDTH + 1];
static unsigned char char_for_xpos_in_state[MAX_STATE][MODE7_WIDTH + 1];
static unsigned char output[MODE7_WIDTH];

static bool global_use_hold = true;
static bool global_use_fill = true;
static bool global_use_sep = true;
static bool global_use_geometric = true;
static bool global_try_all = false;

static int global_sep_fg_factor = 128;

static int frame_width;
static int frame_height;

void clear_error_char_arrays(void)
{
	for (int state = 0; state < MAX_STATE; state++)
	{
		for (int x = 0; x <= MODE7_WIDTH; x++)
		{
			total_error_in_state[state][x] = -1;
			char_for_xpos_in_state[state][x] = 'X';
		}
	}
}

int get_state_for_char(unsigned char proposed_char, int old_state)
{
	int fg = old_state & 7;
	int bg = (old_state >> 3) & 7;
	int hold_mode = (old_state >> 6) & 1;
	unsigned char last_gfx_char = (old_state >> 7) & 0x7f;
	int sep = (old_state >> 14) & 1;

	if (global_use_fill)
	{
		if (proposed_char == MODE7_NEW_BG)
		{
			bg = fg;
		}

		if (proposed_char == MODE7_BLACK_BG)
		{
			bg = 0;
		}
	}

	if (proposed_char > MODE7_GFX_COLOUR && proposed_char < MODE7_GFX_COLOUR + 8)
	{
		fg = proposed_char - MODE7_GFX_COLOUR;
	}

	if (global_use_hold)
	{
		if (proposed_char == MODE7_HOLD_GFX)
		{
			hold_mode = true;
		}

		if (proposed_char == MODE7_RELEASE_GFX)
		{
			hold_mode = false;
			last_gfx_char = MODE7_BLANK;
		}

		if (proposed_char < 128)
		{
			last_gfx_char = proposed_char;
		}
	}
	else
	{
		hold_mode = false;
		last_gfx_char = MODE7_BLANK;
	}

	if (global_use_sep)
	{
		if (proposed_char == MODE7_SEP_GFX)
		{
			sep = true;
		}

		if (proposed_char == MODE7_CONTIG_GFX)
		{
			sep = false;
		}
	}

	return GET_STATE(fg, bg, hold_mode, last_gfx_char, sep);
}


int get_colour_from_rgb(unsigned char r, unsigned char g, unsigned char b)
{
	return (r ? 1 : 0) + (g ? 2 : 0) + (b ? 4 : 0);
}

#define GET_RED_FROM_COLOUR(c)		(c & 1 ? 255:0)
#define GET_GREEN_FROM_COLOUR(c)	(c & 2 ? 255:0)
#define GET_BLUE_FROM_COLOUR(c)		(c & 4 ? 255:0)

unsigned char pixel_to_grey(int mode, unsigned char r, unsigned char g, unsigned char b)
{
	switch (mode)
	{
	case 1:
		return r;

	case 2:
		return g;

	case 3:
		return b;

	case 4:
		return (unsigned char)((r + g + b) / 3);

	case 5:
		return (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);

	default:
		return 0;
	}
}

// For each character cell on this line
// Do we have pixels or not?
// If we have pixels then need to decide whether is it better to replace this cell with a control code or use a graphic character
// If we don't have pixels then need to decide whether it is better to insert a control code or leave empty
// Possible control codes are: new fg colour, fill (bg colour = fg colour), no fill (bg colour = black), hold graphics (hold char = prev char), release graphics (hold char = empty)
// "Better" means that the "error" for the rest of the line (appearance on screen vs actual image = deviation) is minimised

// Hold graphics mode means use last known (used on the line) graphic character in place of space when emitting a control code (reset if using alphanumerics not graphics)
// Palette order = black - red - green - yellow - blue - magenta - cyan - white
// Brightness order = black - blue - red - magenta - green - cyan - yellow - white
// Hue order = red - yellow - green - cyan - blue - magenta - red

// Luma values
// B = 0 ~= 0
// B = 18 = 18 ~= 1x
// R = 54 = 18 + 36 ~= 3x
// M = 73 = 18 + 36 + 19 ~= 4x
// G = 182 = 18 + 36 + 19 + 109 ~= 10x
// C = 201 = 18 + 36 + 19 + 109 + 19 ~= 11x
// Y = 237 = 18 + 36 + 19 + 109 + 19 + 36 ~= 13x
// W = 255 = 18 + 36 + 19 + 109 + 19 + 36 + 18 ~= 14x

static int error_colour_vs_colour[8][8] = {

#if 0		// This maps error to luma when comparing colours against black/white
	{ 0, 3, 10, 13, 1, 4, 11, 14 },		// black
	{ 3, 0, 8, 4, 8, 4, 12, 11 },		// red
	{ 10, 8, 0, 4, 8, 12, 4, 4 },		// green
	{ 13, 4, 4, 0, 12, 8, 8, 1 },		// yellow
	{ 1, 8, 8, 12, 0, 4, 4, 13 },		// blue
	{ 4, 4, 12, 8, 4, 0, 8, 10 },		// magenta
	{ 11, 12, 4, 8, 4, 8, 0, 3 },		// cyan
	{ 14, 11, 4, 1, 13, 10, 3, 0 },		// white
#else		// This maps colours in brightness order when comparing against black/white
	{ 0, 2, 4, 6, 1, 3, 5, 7 },		// black
	{ 2, 0, 4, 2, 4, 2, 6, 5 },		// red
	{ 4, 4, 0, 2, 4, 6, 2, 3 },		// green
	{ 6, 2, 2, 0, 6, 4, 4, 1 },		// yellow
	{ 1, 4, 4, 6, 0, 2, 2, 6 },		// blue
	{ 3, 2, 6, 4, 2, 0, 4, 4 },		// magenta
	{ 5, 6, 2, 4, 2, 4, 0, 2 },		// cyan
	{ 7, 5, 3, 1, 6, 4, 2, 0 },		// white
#endif
};

int error_function(int screen_r, int screen_g, int screen_b, int image_r, int image_g, int image_b)
{
	if (global_use_geometric)
	{
		return (((screen_r - image_r) * (screen_r - image_r)) + ((screen_g - image_g) * (screen_g - image_g)) + ((screen_b - image_b) * (screen_b - image_b))); // / (255 * 255);
	}
	else
	{
		// Use lookup
		return error_colour_vs_colour[get_colour_from_rgb(screen_r, screen_g, screen_b)][get_colour_from_rgb(image_r, image_g, image_b)];
	}
}

int get_error_for_screen_pixel(int x, int y, int screen_bit, int fg, int bg, bool sep)
{
	int screen_r, screen_g, screen_b;
	int image_r, image_g, image_b;

	// These are the pixels that will get written to the screen

	if (screen_bit)
	{
		if (sep)
		{ 
			screen_r = (global_sep_fg_factor * GET_RED_FROM_COLOUR(fg) + (255 - global_sep_fg_factor) * GET_RED_FROM_COLOUR(bg)) / 255;
			screen_g = (global_sep_fg_factor * GET_GREEN_FROM_COLOUR(fg) + (255 - global_sep_fg_factor) * GET_GREEN_FROM_COLOUR(bg)) / 255;
			screen_b = (global_sep_fg_factor * GET_BLUE_FROM_COLOUR(fg) + (255 - global_sep_fg_factor) * GET_BLUE_FROM_COLOUR(bg)) / 255;
		}
		else
		{
			screen_r = GET_RED_FROM_COLOUR(fg);
			screen_g = GET_GREEN_FROM_COLOUR(fg);
			screen_b = GET_BLUE_FROM_COLOUR(fg);
		}
	}
	else
	{
		screen_r = GET_RED_FROM_COLOUR(bg);
		screen_g = GET_GREEN_FROM_COLOUR(bg);
		screen_b = GET_BLUE_FROM_COLOUR(bg);
	}

	// These are the pixels in the image

	image_r = src(x, y, 0);
	image_g = src(x, y, 1);
	image_b = src(x, y, 2);

	// Calculate the error between them

	return error_function(screen_r, screen_g, screen_b, image_r, image_g, image_b);
}

int get_error_for_screen_char(int x7, int y7, unsigned char screen_char, int fg, int bg, bool sep)
{
	int x = IMAGE_X_FROM_X7(x7);
	int y = IMAGE_Y_FROM_Y7(y7);

	int error = 0;

	error += get_error_for_screen_pixel(x, y, screen_char & 1, fg, bg, sep);

	error += get_error_for_screen_pixel(x + 1, y, screen_char & 2, fg, bg, sep);

	error += get_error_for_screen_pixel(x, y + 1, screen_char & 4, fg, bg, sep);

	error += get_error_for_screen_pixel(x + 1, y + 1, screen_char & 8, fg, bg, sep);

	error += get_error_for_screen_pixel(x, y + 2, screen_char & 16, fg, bg, sep);

	error += get_error_for_screen_pixel(x + 1, y + 2, screen_char & 64, fg, bg, sep);

	// For all six pixels in the character cell

	return error;
}

// Functions - get_error_for_char(int x7, int y7, unsigned char code, int fg, int bg, unsigned char hold_char)
int get_error_for_char(int x7, int y7, unsigned char proposed_char, int fg, int bg, bool hold_mode, unsigned char last_gfx_char, bool sep)
{
	// If proposed character >= 128 then this is a control code
	// If so then the hold char will be displayed on screen
	// Otherwise it will be our proposed character (pixels)

	unsigned char screen_char;

	if (hold_mode)
	{
		screen_char = (proposed_char >= 128) ? last_gfx_char : proposed_char;
	}
	else
	{
		screen_char = (proposed_char >= 128) ? MODE7_BLANK : proposed_char;
	}

	return get_error_for_screen_char(x7, y7, screen_char, fg, bg, sep);
}

unsigned char get_graphic_char_from_image(int x7, int y7, int fg, int bg, bool sep)
{
	// Try every possible combination of pixels to get lowest error

	int min_error = INT_MAX;
	unsigned char min_char = 0;

	int x = IMAGE_X_FROM_X7(x7);
	int y = IMAGE_Y_FROM_Y7(y7);

	// This is our default graphics character
	// All pixels matching background are background
	// All other pixels are foreground

	min_char = 32 +																										// bit 5 always set!
			+ (get_colour_from_rgb(src(x, y, 0), src(x, y, 1), src(x, y, 2)) == bg ? 0 : 1)								// (x,y) = bit 0
			+ (get_colour_from_rgb(src(x + 1, y, 0), src(x + 1, y, 1), src(x + 1, y, 2)) == bg ? 0 : 2)					// (x+1,y) = bit 1
			+ (get_colour_from_rgb(src(x, y + 1, 0), src(x, y + 1, 1), src(x, y + 1, 2)) == bg ? 0 : 4)					// (x,y+1) = bit 2
			+ (get_colour_from_rgb(src(x + 1, y + 1, 0), src(x + 1, y + 1, 1), src(x + 1, y + 1, 2)) == bg ? 0 : 8)		// (x+1,y+1) = bit 3
			+ (get_colour_from_rgb(src(x, y + 2, 0), src(x, y + 2, 1), src(x, y + 2, 2)) == bg ? 0 : 16)				// (x,y+2) = bit 4
			+ (get_colour_from_rgb(src(x + 1, y + 2, 0), src(x + 1, y + 2, 1), src(x + 1, y + 2, 2)) == bg ? 0 : 64);	// (x+1,y+2) = bit 6

	// Calculate error for this character

	min_error = get_error_for_screen_char(x7, y7, min_char, fg, bg, sep);

	// See if there's a better character by trying all 64 possible combinations

	for (int i = 1; i < 64; i++)
	{
		unsigned char screen_char = (MODE7_BLANK) | (i & 0x1f) | ((i & 0x20) << 1);

		int error = get_error_for_screen_char(x7, y7, screen_char, fg, bg, sep);

		if (error < min_error)
		{
			min_error = error;
			min_char = screen_char;
		}
	}

	return min_char;
}

int get_error_for_remainder_of_line(int x7, int y7, int fg, int bg, bool hold_mode, unsigned char last_gfx_char, bool sep)
{
	if (x7 >= MODE7_WIDTH)
		return 0;

	int state = GET_STATE(fg, bg, hold_mode, last_gfx_char, sep);

	if (total_error_in_state[state][x7] != -1)
		return total_error_in_state[state][x7];

	//	printf("get_error_for_remainder_of_line(%d, %d, %d, %d, %d, %d)\n", x7, y7, fg, bg, hold_char, prev_char);

	unsigned char graphic_char = global_try_all ? 'Y' : get_graphic_char_from_image(x7, y7, fg, bg, sep);
	int lowest_error = INT_MAX;
	unsigned char lowest_char = 'Z';

	// Possible characters are: 1 + 1 + 6 + 1 + 1 + 1 + 1 = 12 possibilities x 40 columns = 12 ^ 40 combinations.  That's not going to work :)
	// Possible states for a given cell: fg=0-7, bg=0-7, hold_gfx=6 pixels : total = 12 bits = 4096 possible states
	// Wait! What about prev_char as part of state if want to use hold graphics feature? prev_char=6 pixels so actually 18 bits = 262144 possible states
	// Not all of them can be visited as we cannot arbitrarily set the previous character or hold character but still needs a 40Mb array of ints! :S

	// Graphic char (if set)
	// Stay blank (if not)
	// Set graphic colour (colour != fg) x6
	// Fill (if bg != fg)
	// No fill (if bg != 0)
	// Hold graphics (if hold_mode == false)
	// Release graphics (if hold_mode == true)

	// Always try a blank first
	if (MODE7_BLANK)
	{
		int newstate = GET_STATE(fg, bg, hold_mode, MODE7_BLANK, sep);
		int error = get_error_for_char(x7, y7, MODE7_BLANK, fg, bg, hold_mode, MODE7_BLANK, sep);
		int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, bg, hold_mode, MODE7_BLANK, sep);

		if (total_error_in_state[newstate][x7 + 1] == -1)
		{
			total_error_in_state[newstate][x7 + 1] = remaining;
			char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
		}

		error += remaining;

		if (error < lowest_error)
		{
			lowest_error = error;
			lowest_char = MODE7_BLANK;
		}
	}

	// If the background is black we could enable fill! - you idiot - can enable fill at any time if fg colour has changed since last time!
	if (global_use_fill)
	{
		if (bg != fg)
		{
			// Bg colour becomes fg colour immediately in this cell
			int newstate = GET_STATE(fg, fg, hold_mode, last_gfx_char, sep);
			int error = get_error_for_char(x7, y7, MODE7_NEW_BG, fg, fg, hold_mode, last_gfx_char, sep);
			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, fg, hold_mode, last_gfx_char, sep);

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = MODE7_NEW_BG;
			}
		}

		// If the background is not black we could disable fill!
		if (bg != 0)
		{
			// Bg colour becomes black immediately in this cell
			int newstate = GET_STATE(fg, 0, hold_mode, last_gfx_char, sep);
			int error = get_error_for_char(x7, y7, MODE7_BLACK_BG, fg, 0, hold_mode, last_gfx_char, sep);
			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, 0, hold_mode, last_gfx_char, sep);

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = MODE7_BLACK_BG;
			}
		}
	}

	// We could enter seperated graphics mode?
	if (global_use_sep)
	{
		if (!sep)
		{
			int newstate = GET_STATE(fg, bg, hold_mode, last_gfx_char, true);
			int error = get_error_for_char(x7, y7, MODE7_SEP_GFX, fg, bg, hold_mode, last_gfx_char, true);
			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, bg, hold_mode, last_gfx_char, true);

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = MODE7_SEP_GFX;
			}
		}
		
		// We could go back to contiguous graphics...
		else
		{
			int newstate = GET_STATE(fg, bg, hold_mode, last_gfx_char, false);
			int error = get_error_for_char(x7, y7, MODE7_CONTIG_GFX, fg, bg, hold_mode, last_gfx_char, false);
			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, bg, hold_mode, last_gfx_char, false);

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = MODE7_CONTIG_GFX;
			}
		}
	}

	// We could enter hold graphics mode!
	if (global_use_hold)
	{
		if (!hold_mode)
		{
			int newstate = GET_STATE(fg, bg, true, last_gfx_char, sep);
			int error = get_error_for_char(x7, y7, MODE7_HOLD_GFX, fg, bg, true, last_gfx_char, sep);			// hold control code does adopt last graphic character immediately
			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, bg, true, last_gfx_char, sep);

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = MODE7_HOLD_GFX;
			}
		}

		// We could exit hold graphics mode..
		else
		{
			int newstate = GET_STATE(fg, bg, false, MODE7_BLANK, sep);
			int error = get_error_for_char(x7, y7, MODE7_RELEASE_GFX, fg, bg, false, MODE7_BLANK, sep);
			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, bg, false, MODE7_BLANK, sep);

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = MODE7_RELEASE_GFX;
			}
		}
	}

	for (int c = 1; c < 8; c++)
	{
		// We could change our fg colour!
		if (c != fg)
		{
			int newstate = GET_STATE(c, bg, hold_mode, last_gfx_char, sep);

			// The fg colour doesn't actually take effect until next cell - so any hold char here will be in current fg colour
			int error = get_error_for_char(x7, y7, MODE7_GFX_COLOUR + c, fg, bg, hold_mode, last_gfx_char, sep);			// old state

			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, c, bg, hold_mode, last_gfx_char, sep);			// new state

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = MODE7_GFX_COLOUR + c;
			}
		}
	}

	if (global_try_all)
	{
		// Try every possible graphic character...

		for (int i = 1; i < 64; i++)
		{
			graphic_char = (MODE7_BLANK) | (i & 0x1f) | ((i & 0x20) << 1);

			int newstate = GET_STATE(fg, bg, hold_mode, global_use_hold ? graphic_char : MODE7_BLANK, sep);
			int error = get_error_for_char(x7, y7, graphic_char, fg, bg, hold_mode, global_use_hold ? graphic_char : MODE7_BLANK, sep);
			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, bg, hold_mode, global_use_hold ? graphic_char : MODE7_BLANK, sep);

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = graphic_char;
			}
		}
	}
	else
	{
		// Try our graphic character (if it's not blank)

		if (graphic_char != MODE7_BLANK)
		{
			int newstate = GET_STATE(fg, bg, hold_mode, global_use_hold ? graphic_char : MODE7_BLANK, sep);
			int error = get_error_for_char(x7, y7, graphic_char, fg, bg, hold_mode, global_use_hold ? graphic_char : MODE7_BLANK, sep);
			int remaining = get_error_for_remainder_of_line(x7 + 1, y7, fg, bg, hold_mode, global_use_hold ? graphic_char : MODE7_BLANK, sep);

			if (total_error_in_state[newstate][x7 + 1] == -1)
			{
				total_error_in_state[newstate][x7 + 1] = remaining;
				char_for_xpos_in_state[newstate][x7 + 1] = output[x7 + 1];
			}

			error += remaining;

			if (error < lowest_error)
			{
				lowest_error = error;
				lowest_char = graphic_char;
			}
		}
	}
	//	printf("(%d, %d) returning char=%d lowest error=%d\n", x7, y7, lowest_char, lowest_error);

	output[x7] = lowest_char;

	return lowest_error;
}

int match_closest_palette_colour(unsigned char r, unsigned char g, unsigned char b)
{
	int min_error = INT_MAX;
	int min_colour = -1;

	for (int c = 0; c < 8; c++)
	{
		unsigned char cr = GET_RED_FROM_COLOUR(c);
		unsigned char cg = GET_GREEN_FROM_COLOUR(c);
		unsigned char cb = GET_BLUE_FROM_COLOUR(c);

		int error = ((cr - r) * (cr - r)) + ((cg - g) * (cg - g)) + ((cb - b) * (cb - b));

		if (error < min_error)
		{
			min_error = error;
			min_colour = c;
		}
	}

	return min_colour;
}

int main(int argc, char **argv)
{
	cimg_usage("MODE 7 image convertor.\n\nUsage : image2mode7 [options]");
	const char *const input_name = cimg_option("-i", (char*)0, "Input filename");
	const char *const output_name = cimg_option("-o", (char*)0, "Output filename");
	const int sat = cimg_option("-sat", 64, "Saturation threshold (below this colour is considered grey)");
	const int value = cimg_option("-val", 64, "Value threshold (below this colour is considered black)");
	const int black = cimg_option("-black", 64, "Black threshold (grey below this considered pure black - above is colour brightness ramp)");
	const int white = cimg_option("-white", 128, "White threshold (grey above this considered pure white - below is colour brightness ramp)");
	const bool use_quant = cimg_option("-quant", false, "Quantise the input image to 3-bit MODE 7 palette using HSV params above");
	const bool no_hold = cimg_option("-nohold", false, "Disallow Hold Graphics control code");
	const bool no_fill = cimg_option("-nofill", false, "Disallow New Background control code");
	const bool use_sep = cimg_option("-sep", false, "Enable Separated Graphics control code");
	const int sep_factor = cimg_option("-fore", 128, "Contribution factor of foreground vs background colour for separated graphics");
	const bool no_scale = cimg_option("-noscale", false, "Don't scale the image image to MODE 7 resolution");
	const bool simg = cimg_option("-test", false, "Save test images (quantised / scaled) before Teletext conversion");
	const bool inf = cimg_option("-inf", false, "Save inf file for output file");
	const bool verbose = cimg_option("-v", false, "Verbose output");
	const bool url = cimg_option("-url", false, "Spit out URL for edit.tf");
	const bool error_lookup = cimg_option("-lookup", false, "Use lookup table for colour error (default is geometric distance)");
	const bool try_all = cimg_option("-tryall", false, "Calculate full line error for every possible graphics character (64x slower)");

	char filename[256];
	FILE *file;

	if (cimg_option("-h", false, 0)) std::exit(0);
	if (input_name == NULL)  std::exit(0);

	global_use_hold = !no_hold;
	global_use_fill = !no_fill;
	global_use_sep = use_sep;
	global_sep_fg_factor = sep_factor;

	global_use_geometric = !error_lookup;
	global_try_all = try_all;

	if (verbose)
	{
		printf("Loading image file '%s'...\n", input_name);
	}

	src.assign(input_name);

	//
	// Colour conversion etc.
	//

	if (!use_quant)
	{
		if (verbose)
		{
			printf("Skipping conversion to MODE 7 palette...\n");
		}
	}
	else
	{
		if (verbose)
		{
			printf("Converting to MODE 7 palette...\n");
		}

		// Convert to HSV

		cimg_forXY(src, x, y)
		{
			unsigned char R = src(x, y, 0);
			unsigned char G = src(x, y, 1);
			unsigned char B = src(x, y, 2);

			unsigned char r, g, b;
			r = g = b = 0;

			unsigned char M = MAX_3(R, G, B);
			unsigned char m = MIN_3(R, G, B);

			unsigned char C = M - m;				// Chroma - black to white

			unsigned char Hc = 0;					// Hue - as BBC colour palette

			if (C != 0)
			{
				if (M == R)
				{
					int h = 255 * (G - B) / C;

					if (h > 127) Hc = 3;			// yellow
					else if (h < -128) Hc = 5;		// magenta
					else Hc = 1;					// red
				}
				else if (M == G)
				{
					int h = 255 * (B - R) / C;

					if (h > 127) Hc = 6;			// cyan
					else if (h < -128) Hc = 3;		// yellow
					else Hc = 2;					// green
				}
				else if (M == B)
				{
					int h = 255 * (R - G) / C;

					if (h > 127) Hc = 5;			// magenta
					else if (h < -128) Hc = 6;		// cyan
					else Hc = 2;					// blue
				}
			}

			unsigned char Y = (unsigned char)(0.2126f * R + 0.7152f * G + 0.0722f * B);		// Luma (screen brightess)

			unsigned char V = M;					// Value

			int S = 0;								// Saturation

			if (C != 0)
			{
				S = 255 * C / V;
			}

			// If saturation too low assume grey

			if (S < sat)
			{
				// Grey
				// Adjust colour palette for grey scale
				// Map value to colour ramp - change RAMP!

				unsigned char Gc = 0;
				int midpoint = (white - black) / 2;

				if (V < black)
					Gc = 0;
				else if (V < (black + midpoint))
					Gc = 4;			// blue
				else if (V < white)
					Gc = 6;			// cyan
				else
					Gc = 7;			// white		// could use yellow?

				r = GET_RED_FROM_COLOUR(Gc);
				g = GET_GREEN_FROM_COLOUR(Gc);
				b = GET_BLUE_FROM_COLOUR(Gc);
			}
			else
			{
				// Colour
				// If Value is too low then assume black

				if (V < value)
				{
					// Black
					r = g = b = 0;
				}
				else
				{
					// Not black = full colour

					int c = match_closest_palette_colour(R, G, B);

					r = GET_RED_FROM_COLOUR(c);
					g = GET_GREEN_FROM_COLOUR(c);
					b = GET_BLUE_FROM_COLOUR(c);
				}
			}

			src(x, y, 0) = r;
			src(x, y, 1) = g;
			src(x, y, 2) = b;
		}

		//
		// Save output of colour conversion for debug
		//

		if (simg)
		{
			if (verbose)
			{
				printf("Saving test image '%s_quant.png'...\n", input_name);
			}

			sprintf(filename, "%s_quant.png", input_name);
			src.save(filename);
		}
	}

	//
	// Resize!
	//

	int pixel_width, pixel_height;

	if (no_scale)
	{
		if (verbose)
		{
			printf("Leaving size as %d x %d pixels...\n", src._width, src._height);
		}

		pixel_width = src._width;
		pixel_height = src._height;
	}
	else
	{
		// Calculate frame size - adjust to width
		pixel_width = (MODE7_WIDTH - FRAME_FIRST_COLUMN) * 2;
		pixel_height = pixel_width * IMAGE_H / IMAGE_W;
		if (pixel_height % 3) pixel_height += (3 - (pixel_height % 3));

		// Adjust to height
		if (pixel_height > MODE7_PIXEL_H)
		{
			pixel_height = MODE7_PIXEL_H;
			pixel_width = pixel_height * IMAGE_W / IMAGE_H;

			if (pixel_width % 1) pixel_width++;

			// Need to handle reset of background if frame_width < MODE7_WIDTH
		}

		// Resize image to this size

		if (verbose)
		{
			printf("Resizing from %d x %d to %d x %d pixels...\n", src._width, src._height, pixel_width, pixel_height);
		}

		src.resize(pixel_width, pixel_height);

		// Save test images for debug

		if (simg)
		{
			if (verbose)
			{
				printf("Saving test image '%s_small.png'...\n", input_name);
			}

			sprintf(filename, "%s_small.png", input_name);
			src.save(filename);
		}
	}

	//
	// Conversion to MODE 7
	//

	int frame_error = 0;

	frame_width = pixel_width / 2;
	frame_height = pixel_height / 3;

	if (verbose)
	{
		printf("Converting to MODE 7 screen size %d x %d...\n", frame_width, frame_height);
	}

	// Set everything to blank
	memset(mode7, MODE7_BLANK, MODE7_MAX_SIZE);

	for(int y7 = 0; y7 < frame_height; y7++)
	{
		int y = IMAGE_Y_FROM_Y7(y7);

		// Reset state as starting new character row
		// State = fg colour + bg colour + hold character + prev character
		// For each character cell on this line
		// Do we have pixels or not?
		// If we have pixels then need to decide whether is it better to replace this cell with a control code or keep pixels
		// Possible control codes are: new fg colour, fill (bg colour = fg colour), no fill (bg colour = black), hold graphics (hold char = prev char), release graphics (hold char = empty)
		// "Better" means that the "error" for the rest of the line (appearance on screen vs actual image = deviation) is minimised

		// Clear our array of error values for each state & x position
		clear_error_char_arrays();

		int min_error = INT_MAX;
		int min_colour = 0;

		// Determine best initial state for line
		for (int fg = 7; fg > 0; fg--)
		{
			// What would our first character look like in this state?
			unsigned char first_char = get_graphic_char_from_image(FRAME_FIRST_COLUMN, y7, fg, 0, false);

			// What's the error for that character?
			int error = get_error_for_char(FRAME_FIRST_COLUMN, y7, first_char, fg, 0, false, MODE7_BLANK, false);

			// Find the lowest error corresponding to our possible start states
			if (error < min_error)
			{
				min_error = error;
				min_colour = fg;
			}
		}

		// This is our initial state of the line
		int state = GET_STATE(min_colour, 0, false, MODE7_BLANK, false);

		if (verbose)
		{
			printf("[%d] Start colour=%d ", y7, min_colour);
		}

		// Set this state before frame begins
		mode7[(y7 * MODE7_WIDTH) + (FRAME_FIRST_COLUMN - 1)] = MODE7_GFX_COLOUR + min_colour;

		// Kick off recursive error calculation with that state
		int error = get_error_for_remainder_of_line(FRAME_FIRST_COLUMN, y7, min_colour, 0, false, MODE7_BLANK, false);

		if (verbose)
		{
			printf("Line error=%d\n", error);
		}

		frame_error += error;

		// Store first character
		char_for_xpos_in_state[state][FRAME_FIRST_COLUMN] = output[FRAME_FIRST_COLUMN];

		// Copy the resulting character data into MODE 7 screen
		for (int x7 = FRAME_FIRST_COLUMN; x7 < (FRAME_FIRST_COLUMN + FRAME_WIDTH); x7++)
		{
			// Copy character chosen in this position for this state
			unsigned char best_char = char_for_xpos_in_state[state][x7];

			mode7[(y7 * MODE7_WIDTH) + (x7)] = best_char;

			// Update the state
			state = get_state_for_char(best_char, state);
		}

		// For when image is narrower than screen width

		if (FRAME_FIRST_COLUMN + FRAME_WIDTH < MODE7_WIDTH)
		{
			mode7[(y7 * MODE7_WIDTH) + FRAME_FIRST_COLUMN + FRAME_WIDTH] = MODE7_BLACK_BG;
		}

		// printf("\n");

		y += 2;
	}

	if (verbose)
	{
		printf("Total frame error = %d\n", frame_error);
		printf("MODE 7 frame size = %d bytes\n", FRAME_SIZE);
	}

	if (output_name)
	{
		if (verbose)
		{
			printf("Writing MODE 7 frame '%s'...\n", output_name);
		}

		file = fopen(output_name, "wb");
	}
	else
	{
		if (verbose)
		{
			printf("Writing MODE 7 frame '%s.bin'...\n", input_name);
		}

		sprintf(filename, "%s.bin", input_name);
		file = fopen(filename, "wb");
	}

	if (file)
	{
		fwrite(mode7, 1, FRAME_SIZE, file);
		fclose(file);
	}

	if (inf)
	{
		if (output_name)
		{
			if (verbose)
			{
				printf("Writing inf file '%s.inf'...\n", output_name);
			}

			sprintf(filename, "%s.inf", output_name);
		}
		else
		{
			if (verbose)
			{
				printf("Writing inf file '%s.bin.inf'...\n", input_name);
			}

			sprintf(filename, "%s.bin.inf", input_name);
		}

		file = fopen((const char*)filename, "wb");

		if (file)
		{
			char buffer[256];
			sprintf(buffer, "$.IMAGE      FF7C00 FF7C00\n");

			fwrite(buffer, 1, strlen(buffer), file);
			fclose(file);
		}
	}

	if (url)
	{
		/* set up a destination buffer large enough to hold the encoded data */
		unsigned char *mode77 = (unsigned char *)malloc((MODE7_MAX_SIZE * 7) / 8);
		unsigned char *bits7 = mode77;

		if (verbose)
		{
			printf("Calculating edit.tf URL...\n", input_name);
		}

		for (int i = 0; i < MODE7_MAX_SIZE; i+=8)
		{
			// 8 enter, 7 leave
			unsigned char c;

			c = mode7[i];					// 1
			*bits7 = (c & 0x7f) << 1;

			c = mode7[i + 1];				// 2
			*bits7++ |= (c & 0x40) >> 6;
			*bits7 = (c & 0x3f) << 2;

			c = mode7[i + 2];				// 3
			*bits7++ |= (c & 0x60) >> 5;
			*bits7 = (c & 0x1f) << 3;

			c = mode7[i + 3];				// 4
			*bits7++ |= (c & 0x70) >> 4;
			*bits7 = (c & 0x0f) << 4;

			c = mode7[i + 4];				// 5
			*bits7++ |= (c & 0x78) >> 3;
			*bits7 = (c & 0x07) << 5;

			c = mode7[i + 5];				// 6
			*bits7++ |= (c & 0x7c) >> 2;
			*bits7 = (c & 0x03) << 6;

			c = mode7[i + 6];				// 7
			*bits7++ |= (c & 0x7e) >> 1;
			*bits7 = (c & 0x01) << 7;

			c = mode7[i + 7];				// 8
			*bits7++ |= (c & 0x7f);
		}

		char* base64 = (char*)malloc(4 + (MODE7_MAX_SIZE * 7) / 6);
		/* keep track of our encoded position */
		char* c = base64;
		/* store the number of bytes encoded by a single call */
		int cnt = 0;
		/* we need an encoder state */
		base64_encodestate s;

		/*---------- START ENCODING ----------*/
		/* initialise the encoder state */
		base64_init_encodestate(&s);
		/* gather data from the input and send it to the output */
		cnt = base64_encode_block((const char *)mode77, (MODE7_MAX_SIZE * 7) / 8, c, &s);
		c += cnt;
		/* since we have encoded the entire input string, we know that
		there is no more input data; finalise the encoding */
		cnt = base64_encode_blockend(c, &s);
		c += cnt;
		/*---------- STOP ENCODING  ----------*/

		/* we want to print the encoded data, so null-terminate it: */
		*c = 0;

		printf("http://edit.tf/#0:%s\n", base64);

		free(base64);
		free(mode77);
	}
	
	return 0;
}
