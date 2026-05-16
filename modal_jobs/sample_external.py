"""Sample model on out-of-distribution images.

Reads /data/external_test/*.{png,jpg,jpeg,gif,bmp}, for each one:
  - run image2teletext.convert() to get a DP-solver "gold" reference
  - run model.generate() to get the predicted 1000-byte page
  - render both back to PNG via teletext_decode.render_bytes
Writes /data/<out_dir>/{stem}_{input,pred,gold}.png + index.html.

Use this to measure model quality on images that weren't in training.
"""
import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_FILES = ["image2teletext.py", "teletext_decode.py"]
MODAL_FILES = ["modal_jobs/model.py"]

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "torchvision==0.20.1",
    "transformers==4.46.3", "accelerate==1.1.1",
    "pillow==11.0.0", "numpy==2.1.3",
)
for _f in REPO_FILES:
    image = image.add_local_file(str(REPO_ROOT / _f), f"/repo/{_f}")
for _f in MODAL_FILES:
    image = image.add_local_file(str(REPO_ROOT / _f),
                                 f"/repo/{Path(_f).name}")

volume = modal.Volume.from_name("teletext-m1", create_if_missing=True)
app    = modal.App("teletext-m1-sample-ext", image=image)
DATA   = "/data"


@app.function(volumes={DATA: volume}, gpu="A10G", timeout=30 * 60)
def sample_external(ckpt: str = "ckpt-m3-03.pt",
                    src_dir: str = "external_test",
                    out_dir: str = "samples_external_m3"):
    import sys, os, glob
    sys.path.insert(0, "/repo")
    import numpy as np
    import torch
    from PIL import Image
    from model import build_model, BOS_ID, EOS_ID, PAD_ID, IMAGE_HW, PAGE_LEN
    import teletext_decode as td
    from image2teletext import convert

    src  = f"{DATA}/{src_dir}"
    dst  = f"{DATA}/{out_dir}"
    ckpt_path = f"{DATA}/{ckpt}"
    volume.reload()
    os.makedirs(dst, exist_ok=True)

    print(f"[sample_ext] loading {ckpt_path}", flush=True)
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = build_model().to("cuda").eval()
    model.load_state_dict(ck["model"])

    paths = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
                "*.gif", "*.GIF", "*.bmp", "*.BMP"):
        paths.extend(glob.glob(os.path.join(src, ext)))
    paths.sort()
    print(f"[sample_ext] {len(paths)} images in {src}", flush=True)

    rows = []
    for i, p in enumerate(paths):
        stem = f"{i:02d}_{Path(p).stem}"
        img = Image.open(p).convert("RGB").resize(
            (IMAGE_HW[1], IMAGE_HW[0]), Image.LANCZOS)

        # DP-solver gold
        gold_bytes = bytes(convert(img))
        gold_img = td.render_bytes(gold_bytes)

        # Model pred
        px = (np.asarray(img).astype(np.float32) / 127.5 - 1.0)
        px = torch.from_numpy(px).permute(2, 0, 1).unsqueeze(0).to("cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda",
                                              dtype=torch.bfloat16):
            gen = model.generate(
                pixel_values=px,
                max_length=PAGE_LEN + 1,
                min_length=PAGE_LEN + 1,
                do_sample=False,
                num_beams=1,
                bos_token_id=BOS_ID,
                eos_token_id=None,
                pad_token_id=PAD_ID,
            )
        pred = gen[0, 1:1 + PAGE_LEN].cpu().numpy().astype(np.uint8)
        pred = np.where(pred < 256, pred, 32).astype(np.uint8)
        pred_img = td.render_bytes(bytes(pred))

        img.save(os.path.join(dst, f"{stem}_input.png"))
        pred_img.save(os.path.join(dst, f"{stem}_pred.png"))
        gold_img.save(os.path.join(dst, f"{stem}_gold.png"))

        gold_arr = np.frombuffer(gold_bytes, dtype=np.uint8)
        n_match = int((pred == gold_arr).sum())
        rows.append((stem, Path(p).name, n_match))
        print(f"  {stem}  byte_match_vs_gold={n_match}/1000", flush=True)

    html = ["<html><body style='background:#222;color:#eee;font-family:monospace'>"]
    html.append(f"<h2>external sample — ckpt={ckpt}</h2>")
    html.append("<table cellpadding=8>")
    html.append("<tr><th>name</th><th>input</th><th>pred</th>"
                "<th>DP gold</th><th>match</th></tr>")
    for stem, name, m in rows:
        html.append(
            f"<tr><td>{name}</td>"
            f"<td><img src='{stem}_input.png' width=256></td>"
            f"<td><img src='{stem}_pred.png' width=256></td>"
            f"<td><img src='{stem}_gold.png' width=256></td>"
            f"<td>{m}/1000</td></tr>")
    html.append("</table></body></html>")
    with open(os.path.join(dst, "index.html"), "w") as f:
        f.write("\n".join(html))

    volume.commit()
    avg = sum(m for _, _, m in rows) / max(1, len(rows))
    return {"n": len(rows), "avg_byte_match_vs_dp": round(avg, 1),
            "out_dir": dst}


@app.local_entrypoint()
def main(ckpt: str = "ckpt-m3-03.pt",
         src_dir: str = "external_test",
         out_dir: str = "samples_external_m3"):
    print(sample_external.remote(ckpt, src_dir, out_dir))
