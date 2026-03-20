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
)

def _apply_preset(preset_name):
    p = {**_PRESET_DEFAULTS, **m.PRESETS.get(preset_name, {})}
    return (
        p['par'], p['gamma'], p['contrast'], p['saturation'],
        p['sharpen_amount'], p['sharpen_radius'], p['sharpen_threshold'],
    )

# ---------------------------------------------------------------------------
# Processing preview  (cheap — no solver)
# ---------------------------------------------------------------------------

# Scale factor to display the tiny sub-pixel image at a comfortable size.
# 78 px × 8 = 624 px wide; 75 px × 8 = 600 px tall.
_PREVIEW_SCALE = 8

def _preprocess_inputs(image_path, par, gamma, contrast, saturation,
                       sharpen_amount, sharpen_radius, sharpen_threshold,
                       filter_name, dither, quant_colors):
    """Shared arg list used by both preprocess_preview and convert."""
    return dict(
        filter=filter_name, par=par,
        sharpen_radius=sharpen_radius,
        sharpen_amount=int(sharpen_amount),
        sharpen_threshold=int(sharpen_threshold),
        gamma=gamma, contrast=contrast, saturation=saturation,
        dither=dither,
        quant_colors=int(quant_colors),
    )

def preprocess_preview(image_path, par, gamma, contrast, saturation,
                       sharpen_amount, sharpen_radius, sharpen_threshold,
                       filter_name, dither, quant_colors):
    if image_path is None:
        raise gr.Error("Upload an image first.")
    kwargs = _preprocess_inputs(
        image_path, par, gamma, contrast, saturation,
        sharpen_amount, sharpen_radius, sharpen_threshold,
        filter_name, dither, quant_colors,
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
            filter_name, dither, quant_colors,
            greedy, refine, luma, linear, sep):
    if image_path is None:
        raise gr.Error("Upload an image first.")

    kwargs = _preprocess_inputs(
        image_path, par, gamma, contrast, saturation,
        sharpen_amount, sharpen_radius, sharpen_threshold,
        filter_name, dither, quant_colors,
    )
    page = m.convert_image(
        image_path,
        use_hold=True, use_fill=True, use_sep=sep,
        greedy=greedy, luma=luma, linear=linear, refine=refine,
        **kwargs,
    )

    # Processing preview (reuse the already-run pipeline result)
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
            )

            dither_cb = gr.Checkbox(True,
                label="Dither  (Floyd-Steinberg error diffusion)",
                info="Quantises each sub-pixel to the nearest of the 8 Teletext colours, "
                     "then spreads the colour error to neighbouring pixels. "
                     "This simulates colours that aren't in the palette — e.g. orange, pink, "
                     "or dark shades — by mixing adjacent blocks of red/yellow or "
                     "magenta/red. Works best with natural images and higher saturation.",
            )

            _tv = {**_PRESET_DEFAULTS, **m.PRESETS['tv']}

            with gr.Accordion("Tone & colour", open=True):
                quant_s = gr.Slider(
                    0, 64, value=0, step=1,
                    label="Palette size  (0 = off)",
                    info="Pre-quantise to N colours after resize. Reduces colour "
                         "complexity, producing larger flat regions and a cleaner "
                         "output. Try 8–16 for photos, 4–8 for graphics.",
                )
                gamma_s      = gr.Slider(0.2, 4.0, value=_tv['gamma'],      step=0.05, label="Gamma  (>1 brightens)")
                contrast_s   = gr.Slider(0.0, 4.0, value=_tv['contrast'],   step=0.05, label="Contrast")
                saturation_s = gr.Slider(0.0, 5.0, value=_tv['saturation'], step=0.1,  label="Saturation  (1.5-2.0 recommended)")

            with gr.Accordion("Sharpening", open=False):
                gr.Markdown(
                    "Unsharp mask applied after resize.  "
                    "**Amount** 0 = off.  100–200 suits photos; 300+ suits graphics."
                )
                sharpen_amount_s    = gr.Slider(0,   500, value=_tv['sharpen_amount'],    step=10,  label="Amount %")
                sharpen_radius_s    = gr.Slider(0.1, 5.0, value=_tv['sharpen_radius'],    step=0.1, label="Radius (px)")
                sharpen_threshold_s = gr.Slider(0,   20,  value=_tv['sharpen_threshold'], step=1,   label="Threshold (0 = sharpen everywhere)")

            with gr.Accordion("Display & resize", open=False):
                par_s = gr.Slider(
                    0.5, 2.0, value=1.2, step=0.01,
                    label="Pixel aspect ratio  (1.2 = LCD TV default · 1.0 = emulator · 1.22 = CRT)",
                )
                filter_dd = gr.Dropdown(
                    choices=["bilinear", "lanczos", "bicubic", "nearest", "cimg"],
                    value="bilinear",
                    label="Resize filter",
                )

            with gr.Accordion("Advanced flags", open=False):
                with gr.Row():
                    greedy_cb = gr.Checkbox(False, label="--greedy  (fast, ~4× speedup)")
                    refine_cb = gr.Checkbox(False, label="--refine  (local search on top of DP)")
                with gr.Row():
                    luma_cb   = gr.Checkbox(False, label="--luma  (perceptual error weighting)")
                    linear_cb = gr.Checkbox(False, label="--linear  (sRGB linearisation)")
                with gr.Row():
                    sep_cb    = gr.Checkbox(False, label="--sep  (separated graphics)")

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
                        type="pil",
                    )

                with gr.Tab("Teletext output", id="teletext"):
                    teletext_out = gr.Image(
                        label="Teletext preview  (3× zoom)",
                        type="pil",
                    )
                    url_out = gr.HTML()
                    bin_out = gr.File(label="Download .bin  (load at &7C00 on BBC Micro)")

    # ── Preset → sliders ──────────────────────────────────────────────────
    _slider_outputs = [
        par_s, gamma_s, contrast_s, saturation_s,
        sharpen_amount_s, sharpen_radius_s, sharpen_threshold_s,
    ]
    preset_dd.change(_apply_preset, inputs=[preset_dd], outputs=_slider_outputs)

    # ── Shared input list for preprocessing params ─────────────────────────
    _proc_inputs = [
        image_input,
        par_s, gamma_s, contrast_s, saturation_s,
        sharpen_amount_s, sharpen_radius_s, sharpen_threshold_s,
        filter_dd, dither_cb, quant_s,
    ]

    _conv_inputs = _proc_inputs + [
        greedy_cb, refine_cb, luma_cb, linear_cb, sep_cb,
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
    image_input.change(
        preprocess_preview,
        inputs=_proc_inputs,
        outputs=[processed_out, output_tabs],
    )


if __name__ == "__main__":
    demo.launch()
