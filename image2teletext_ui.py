#!/usr/bin/env python3
"""
image2teletext_ui.py — Gradio web UI for image2teletext.py

Run:  python image2teletext_ui.py
Then open http://localhost:7860 in your browser.

Deploy to Hugging Face Spaces later by uploading:
  image2teletext_ui.py  (rename to app.py)
  image2teletext.py
  requirements.txt
"""

import sys
import tempfile
sys.path.insert(0, '.')

import gradio as gr
from PIL import Image
import image2teletext as m

# ---------------------------------------------------------------------------
# Preset → slider update
# ---------------------------------------------------------------------------

_PRESET_DEFAULTS = dict(
    par=1.2, gamma=1.0, contrast=1.0, saturation=1.0,
    sharpen_amount=0, sharpen_radius=1.0, sharpen_threshold=0,
    snap=0, quant_colors=0, posterize=0, median=0, snap_palette=False,
    bg_flatten=0, edge_weight=1.0,
)

def _apply_preset(preset_name):
    p = {**_PRESET_DEFAULTS, **m.PRESETS.get(preset_name, {})}
    return (
        p['par'], p['gamma'], p['contrast'], p['saturation'],
        p['sharpen_amount'], p['sharpen_radius'], p['sharpen_threshold'],
        p['snap'], p['quant_colors'], p['posterize'], p['median'],
        p['snap_palette'], p['bg_flatten'], p['edge_weight'],
    )

# ---------------------------------------------------------------------------
# Processing preview  (cheap — no solver)
# ---------------------------------------------------------------------------

# Scale factor to display the tiny sub-pixel image at a comfortable size.
# 78 px × 8 = 624 px wide; 75 px × 8 = 600 px tall.
_PREVIEW_SCALE = 8

def _preprocess_inputs(image_path, par, gamma, contrast, saturation,
                       sharpen_amount, sharpen_radius, sharpen_threshold,
                       filter_name, dither, quant_colors, posterize, snap, median,
                       snap_palette, bg_flatten, direct_sample):
    """Shared arg list used by both preprocess_preview and convert."""
    return dict(
        filter=filter_name, par=par,
        sharpen_radius=sharpen_radius,
        sharpen_amount=int(sharpen_amount),
        sharpen_threshold=int(sharpen_threshold),
        gamma=gamma, contrast=contrast, saturation=saturation,
        dither=dither,
        quant_colors=int(quant_colors),
        posterize=int(posterize),
        snap=int(snap),
        median=int(median),
        snap_palette=bool(snap_palette),
        bg_flatten=int(bg_flatten),
        direct_sample=bool(direct_sample),
    )

def preprocess_preview(image_path, par, gamma, contrast, saturation,
                       sharpen_amount, sharpen_radius, sharpen_threshold,
                       filter_name, dither, quant_colors, posterize, snap, median,
                       snap_palette, bg_flatten, direct_sample):
    if image_path is None:
        raise gr.Error("Upload an image first.")
    kwargs = _preprocess_inputs(
        image_path, par, gamma, contrast, saturation,
        sharpen_amount, sharpen_radius, sharpen_threshold,
        filter_name, dither, quant_colors, posterize, snap, median, snap_palette, bg_flatten,
        direct_sample,
    )
    img = m.preprocess_image(image_path, **kwargs)
    # Scale up so individual sub-pixels are clearly visible
    return img.resize(
        (img.width * _PREVIEW_SCALE, img.height * _PREVIEW_SCALE),
        resample=Image.NEAREST,
    ), gr.update(selected="processed")

# ---------------------------------------------------------------------------
# Full conversion
# ---------------------------------------------------------------------------

def convert(image_path, par, gamma, contrast, saturation,
            sharpen_amount, sharpen_radius, sharpen_threshold,
            filter_name, dither, quant_colors, posterize, snap, median,
            snap_palette, bg_flatten, direct_sample,
            greedy, refine, luma, linear, sep, smooth, edge_weight):
    if image_path is None:
        raise gr.Error("Upload an image first.")

    kwargs = _preprocess_inputs(
        image_path, par, gamma, contrast, saturation,
        sharpen_amount, sharpen_radius, sharpen_threshold,
        filter_name, dither, quant_colors, posterize, snap, median, snap_palette, bg_flatten,
        direct_sample,
    )
    page = m.convert_image(
        image_path,
        use_hold=True, use_fill=True, use_sep=sep,
        greedy=greedy, luma=luma, linear=linear, refine=refine,
        smooth=int(smooth),
        edge_weight=float(edge_weight),
        **kwargs,
    )

    # Processing preview — rerun preprocess (kwargs already has direct_sample)
    proc_img = m.preprocess_image(image_path, **kwargs)
    proc_img = proc_img.resize(
        (proc_img.width * _PREVIEW_SCALE, proc_img.height * _PREVIEW_SCALE),
        resample=Image.NEAREST,
    )

    # Teletext preview — 3× zoom
    teletext_img = m.render_preview(page)
    teletext_img = teletext_img.resize(
        (teletext_img.width * 3, teletext_img.height * 3), resample=Image.NEAREST
    )

    # URLs
    edittf = m.to_edittf_url(page)
    zxnet  = m.to_zxnet_url(page)
    _btn = ("display:inline-block;padding:8px 16px;border-radius:6px;font-weight:600;"
            "text-decoration:none;color:#fff;background:#2563eb;margin-right:8px")
    url_html = (f'<a href="{edittf}" target="_blank" style="{_btn}">Open in edit.tf</a>'
                f'<a href="{zxnet}"  target="_blank" style="{_btn}">Open in ZXNet</a>')

    # .bin download
    tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    tmp.write(bytes(page))
    tmp.close()

    return proc_img, teletext_img, url_html, tmp.name, gr.update(selected="teletext")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_preprocess_params = None   # filled in after widget creation

with gr.Blocks(title="image2teletext") as demo:

    gr.Markdown(
        "# image2teletext\n"
        "Convert an image to **Teletext / BBC Micro Mode 7** graphics "
        "(40×25 characters, 1000 bytes).  "
        "Choose a preset to get started, then tweak the sliders."
    )

    with gr.Row():

        # ── Left column: controls ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=360):

            image_input = gr.Image(
                type="filepath", label="Input image",
                sources=["upload", "clipboard"],
            )

            preset_dd = gr.Dropdown(
                choices=[""] + sorted(m.PRESETS.keys()),
                value="tv",
                label="Preset  (snaps sliders; you can still override)",
                allow_custom_value=False,
                info="Named combinations of settings for common use cases. "
                     "photo: balanced portraits and landscapes; "
                     "clean: like photo but with light denoising and colour snap for less speckle; "
                     "smooth: heavy noise reduction, best for noisy JPEGs and soft gradients; "
                     "vivid: punchy colours, strong edges; "
                     "graphic: flat-colour logos and cartoons; "
                     "flat: bold posterised look with limited palette; "
                     "retro: authentic Ceefax style with blocky limited colours; "
                     "art: teletext-art style — quantise to 6 colours then snap each region to Teletext; "
                     "dark: underexposed images (gamma lift); "
                     "tv: LCD TV viewing (PAR 1.2); "
                     "crt: original CRT viewing (PAR 1.22).",
            )

            _tv = {**_PRESET_DEFAULTS, **m.PRESETS['tv']}

            with gr.Accordion("Image processing", open=True):
                gr.Markdown(
                    "Standard image adjustments applied **before** the Teletext solver runs."
                )
                gamma_s = gr.Slider(0.2, 4.0, value=_tv['gamma'], step=0.05,
                    label="Gamma  (>1 brightens)",
                    info="Power-law tone adjustment applied before resize. "
                         "Values >1 lift shadows and brighten the image; <1 darken it. "
                         "1.5–2.2 suits dark or underexposed photos. "
                         "1.0 = off.",
                )
                contrast_s = gr.Slider(0.0, 4.0, value=_tv['contrast'], step=0.05,
                    label="Contrast",
                    info="Pushes darks darker and lights lighter. "
                         "1.0 = unchanged; 1.2–1.5 suits most photos; "
                         "higher values clip highlights and shadows. "
                         "Applied before resize.",
                )
                saturation_s = gr.Slider(0.0, 5.0, value=_tv['saturation'], step=0.1,
                    label="Saturation  (1.5–2.0 recommended)",
                    info="The Teletext palette is fully saturated — every channel is "
                         "either 0 or 255. Boosting saturation pushes source colours "
                         "toward the palette, reducing dithering artefacts. "
                         "0 = greyscale; 1.0 = unchanged; 1.5–2.0 recommended for photos.",
                )
                median_s = gr.Slider(
                    0, 7, value=0, step=1,
                    label="Denoise  (0 = off · 1 = 3×3 · 2 = 5×5 · 3 = 7×7)",
                    info="Median filter: smooths noise and small colour blobs before resize "
                         "while preserving hard edges. Better than blur for photos with "
                         "fine detail. Good for reducing JPEG artefacts and output speckle.",
                )
                sharpen_amount_s = gr.Slider(0, 500, value=_tv['sharpen_amount'], step=10,
                    label="Sharpen amount %  (0 = off)",
                    info="Unsharp mask strength applied after resize. Pushes pixel values "
                         "toward channel extremes, improving Teletext palette matching. "
                         "100–200 for photos; 300+ for graphics and logos.",
                )
                with gr.Row():
                    sharpen_radius_s = gr.Slider(0.1, 5.0, value=_tv['sharpen_radius'], step=0.1,
                        label="Sharpen radius (px)",
                        info="Spatial extent of the unsharp mask, in sub-pixels. "
                             "0.5 = sub-pixel edges only; 1.0 = one character cell (recommended); "
                             "2.0+ widens halos, useful for enhancing broad colour regions.",
                    )
                    sharpen_threshold_s = gr.Slider(0, 20, value=_tv['sharpen_threshold'], step=1,
                        label="Sharpen threshold",
                        info="Minimum per-channel difference before sharpening is applied. "
                             "0 = sharpen all pixels including smooth gradients; "
                             "3–10 = skip near-flat areas, avoids amplifying JPEG noise; "
                             "10–20 = pronounced edges only.",
                    )

            with gr.Accordion("Teletext palette", open=True):
                gr.Markdown(
                    "Controls how colours are reduced to the 8-colour Teletext palette."
                )
                dither_cb = gr.Checkbox(True,
                    label="Dither  (Floyd-Steinberg error diffusion)",
                    info="Quantises each sub-pixel to the nearest of the 8 Teletext colours, "
                         "then spreads the colour error to neighbouring pixels. "
                         "Simulates colours not in the palette — e.g. orange, pink, dark shades "
                         "— by mixing adjacent blocks. Works best with higher saturation.",
                )
                bg_flatten_s = gr.Slider(
                    0, 128, value=0, step=1,
                    label="Background flatten  (0 = off)",
                    info="Detect the dominant background colour from the image border, "
                         "then replace all pixels within this Euclidean RGB distance of "
                         "that colour with the nearest Teletext colour. "
                         "40 = conservative; 60 = moderate; 80+ = aggressive. "
                         "Works best on images with a plain, uniform background.",
                )
                quant_s = gr.Slider(
                    0, 64, value=0, step=1,
                    label="Palette size  (0 = off)",
                    info="Pre-quantise to N colours after resize. Reduces colour "
                         "complexity, producing larger flat regions and a cleaner "
                         "output. Try 8–16 for photos, 4–8 for graphics.",
                )
                snap_s = gr.Slider(
                    0, 128, value=0, step=1,
                    label="Colour snap tolerance  (0 = off)",
                    info="Snap pixels within this Euclidean RGB distance of a Teletext "
                         "palette colour to that colour, before dithering. Reduces "
                         "ambiguous mid-tones that cause noisy or speckled output. "
                         "Try 20–40 for subtle snapping; 60–80 for stronger effect.",
                )
                with gr.Row():
                    snap_palette_cb = gr.Checkbox(False,
                        label="Snap palette to Teletext colours",
                        info="After quantising, map every colour region unconditionally "
                             "to its nearest Teletext colour. Produces smooth flat regions "
                             "in pure Teletext colours — the 'art palette' look of hand-crafted "
                             "teletext. Best used with Palette size 4–8.",
                    )
                    posterize_s = gr.Slider(
                        0, 7, value=0, step=1,
                        label="Posterise  (0 = off, bits per channel)",
                        info="Snap each colour channel to 2^N evenly spaced values before "
                             "resize. 1 = only 0/255 per channel (8 colours); "
                             "2 = 4 steps; 3–4 suits most photos. "
                             "Creates bold flat regions with hard edges.",
                    )

            with gr.Accordion("Teletext encoding", open=False):
                gr.Markdown(
                    "Options specific to the Mode 7 character solver and display target."
                )
                with gr.Row():
                    par_s = gr.Slider(
                        0.5, 2.0, value=1.2, step=0.01,
                        label="Pixel aspect ratio",
                        info="1.2 = modern LCD TV (default); 1.0 = emulator / square pixels; "
                             "1.22 = original CRT. The SAA5050 displays characters taller than "
                             "wide on a real TV; PAR > 1 pre-squishes the image so the display "
                             "stretches it back correctly.",
                    )
                    filter_dd = gr.Dropdown(
                        choices=["bilinear", "lanczos", "bicubic", "nearest", "cimg"],
                        value="bilinear",
                        label="Resize filter",
                        info="Resampling method for scaling down to the 78×75 sub-pixel canvas. "
                             "Bilinear = fast and smooth; Lanczos = sharper; "
                             "Nearest = hard pixel edges; CImg = matches original C++ tool.",
                    )
                edge_weight_s = gr.Slider(
                    1.0, 8.0, value=1.0, step=0.5,
                    label="Edge weight  (1.0 = off · silhouette priority)",
                    info="Scale the solver error for cells at strong colour boundaries "
                         "by up to this factor. Prioritises faithful silhouette reproduction "
                         "at the cost of some accuracy in flat uniform regions. "
                         "1.0 = off; 2–3 = subtle; 4–5 = strong. "
                         "Pairs well with Separated graphics.",
                )
                smooth_s = gr.Slider(
                    0, 8, value=0, step=1,
                    label="Colour run smoothing  (0 = off)",
                    info="After solving, merge colour runs shorter than N cells into "
                         "the dominant neighbouring colour and re-render. Eliminates "
                         "the salt-and-pepper colour noise typical of automated "
                         "conversion. 2–3 = subtle; 4–6 = bolder, more hand-drawn.",
                )
                with gr.Row():
                    sep_cb = gr.Checkbox(False,
                        label="Separated graphics",
                        info="Enable separated graphics mode (--sep). Each sub-pixel block "
                             "has a 1-pixel gap, giving a more open gridded look. "
                             "Pairs well with Edge weight > 1 for textured areas.",
                    )
                    direct_sample_cb = gr.Checkbox(False,
                        label="Direct sample  (Teletext sources)",
                        info="Bypass bilinear resize (--direct-sample): quantise at full "
                             "resolution and point-sample at the exact sub-pixel grid. "
                             "Preserves fine patterns that bilinear resize blurs. "
                             "Ideal for images that are already Teletext renders.",
                    )
                with gr.Row():
                    greedy_cb = gr.Checkbox(False,
                        label="Greedy solver  (~4× faster)",
                        info="Use a fast left-to-right greedy solver (--greedy) instead of "
                             "full dynamic programming. Good for quick previews.",
                    )
                    refine_cb = gr.Checkbox(False,
                        label="Refine  (local search after DP)",
                        info="After the main solve, re-try every character with all valid "
                             "alternatives and accept improvements until convergence (--refine). "
                             "Slow but recovers a few percent of quality.",
                    )
                with gr.Row():
                    luma_cb = gr.Checkbox(False,
                        label="Luma weighting",
                        info="Weight colour errors by ITU-R BT.601 luminance (--luma). "
                             "Reduces blue fringing in favour of correct perceived brightness.",
                    )
                    linear_cb = gr.Checkbox(False,
                        label="Linear light error",
                        info="Linearise source pixels from sRGB before computing error (--linear). "
                             "Corrects for gamma encoding; dark-tone differences are weighted "
                             "more fairly.",
                    )

            with gr.Row():
                preview_btn = gr.Button("Preview processing", variant="secondary")
                convert_btn = gr.Button("Convert to Teletext", variant="primary")

        # ── Right column: outputs ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=380):

            with gr.Tabs(selected="processed") as output_tabs:
                with gr.Tab("Processed input", id="processed"):
                    gr.Markdown(
                        "How your image looks after tone/colour, sharpening and resize — "
                        "**before** the Teletext character solver runs.  "
                        "Each pixel here is one Teletext sub-pixel (2×3 per character cell).  "
                        "Click **Preview processing** to update without waiting for the full conversion."
                    )
                    processed_out = gr.Image(
                        label=f"Sub-pixel canvas ({_PREVIEW_SCALE}× zoom)",
                        type="pil", format="png",
                    )

                with gr.Tab("Teletext output", id="teletext"):
                    teletext_out = gr.Image(
                        label="Teletext preview  (3× zoom)",
                        type="pil", format="png",
                    )
                    url_out = gr.HTML()
                    bin_out = gr.File(label="Download .bin  (load at &7C00 on BBC Micro)")

    # ── Preset → sliders ──────────────────────────────────────────────────
    _slider_outputs = [
        par_s, gamma_s, contrast_s, saturation_s,
        sharpen_amount_s, sharpen_radius_s, sharpen_threshold_s,
        snap_s, quant_s, posterize_s, median_s, snap_palette_cb, bg_flatten_s,
        edge_weight_s,
    ]
    preset_dd.change(_apply_preset, inputs=[preset_dd], outputs=_slider_outputs)

    # ── Shared input list for preprocessing params ─────────────────────────
    _proc_inputs = [
        image_input,
        par_s, gamma_s, contrast_s, saturation_s,
        sharpen_amount_s, sharpen_radius_s, sharpen_threshold_s,
        filter_dd, dither_cb, quant_s, posterize_s, snap_s, median_s,
        snap_palette_cb, bg_flatten_s, direct_sample_cb,
    ]

    _conv_inputs = _proc_inputs + [
        greedy_cb, refine_cb, luma_cb, linear_cb, sep_cb, smooth_s, edge_weight_s,
    ]

    # ── Preview button ────────────────────────────────────────────────────
    preview_btn.click(
        preprocess_preview,
        inputs=_proc_inputs,
        outputs=[processed_out, output_tabs],
    )

    # ── Convert button — updates both tabs ───────────────────────────────
    convert_btn.click(
        convert,
        inputs=_conv_inputs,
        outputs=[processed_out, teletext_out, url_out, bin_out, output_tabs],
    )

    # ── Auto-update processing preview on image upload ────────────────────
    # Use .upload (not .change) so this only fires when an image is fully
    # uploaded — not during the transient clear that occurs when replacing
    # an existing image with a new one.
    image_input.upload(
        preprocess_preview,
        inputs=_proc_inputs,
        outputs=[processed_out, output_tabs],
    )


if __name__ == "__main__":
    demo.launch()
