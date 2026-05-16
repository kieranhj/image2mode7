"""Stage A sanity check — sample from ckpt-01.pt on gold images.

For each of N gold horsenburger images:
  - resize to 256x192
  - run model.generate() to get predicted 1000-byte page
  - render predicted page back to PNG via teletext_decode.render_bytes
  - render gold (true) page the same way for comparison

Saves a /data/samples/ folder with input.png, pred.png, gold.png per sample
plus an index.html grid for easy browsing.
"""
import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_FILES = [
    "image2teletext.py",
    "teletext_decode.py",
]
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
app    = modal.App("teletext-m1-sample", image=image)
DATA   = "/data"

GALLERY_DIR = f"{DATA}/horsenburger"
GOLD_NPZ    = f"{DATA}/gold.npz"
CKPT_PATH   = f"{DATA}/ckpt-m3-03.pt"
OUT_DIR     = f"{DATA}/samples_m3"


@app.function(volumes={DATA: volume}, gpu="A10G", timeout=20 * 60)
def sample(n: int = 12, seed: int = 0):
    import sys, os, glob, io
    sys.path.insert(0, "/repo")
    import numpy as np
    import torch
    from PIL import Image
    from model import build_model, BOS_ID, EOS_ID, PAD_ID, IMAGE_HW, PAGE_LEN
    import teletext_decode as td

    volume.reload()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[sample] loading ckpt {CKPT_PATH}", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
    model = build_model().to("cuda").eval()
    model.load_state_dict(ckpt["model"])

    gold = np.load(GOLD_NPZ)
    names = list(gold["names"])
    bytes_arr = gold["bytes"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(names), size=min(n, len(names)), replace=False)
    print(f"[sample] selected {len(pick)} images", flush=True)

    rows = []
    for i, idx in enumerate(pick):
        name = str(names[idx])
        png_path = os.path.join(GALLERY_DIR, name)
        if not os.path.exists(png_path):
            print(f"  skip missing {name}", flush=True)
            continue

        img = Image.open(png_path).convert("RGB").resize(
            (IMAGE_HW[1], IMAGE_HW[0]), Image.LANCZOS)
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
        # gen[0] = [BOS, b0, b1, ..., b999]
        pred = gen[0, 1:1 + PAGE_LEN].cpu().numpy().astype(np.uint8)
        # Clamp anything in the special-token range back to a printable byte.
        pred = np.where(pred < 256, pred, 32).astype(np.uint8)

        true_bytes = bytes_arr[idx]

        # Render (render_bytes returns a PIL Image)
        pred_img = td.render_bytes(bytes(pred))
        gold_img = td.render_bytes(bytes(true_bytes))

        stem = f"{i:02d}_{Path(name).stem}"
        img.save(os.path.join(OUT_DIR, f"{stem}_input.png"))
        pred_img.save(os.path.join(OUT_DIR, f"{stem}_pred.png"))
        gold_img.save(os.path.join(OUT_DIR, f"{stem}_gold.png"))

        n_match = int((pred == true_bytes).sum())
        rows.append((stem, name, n_match))
        print(f"  {stem}  byte_match={n_match}/1000", flush=True)

    html = ["<html><body style='background:#222;color:#eee;font-family:monospace'>"]
    html.append("<h2>ckpt-01 sanity check</h2>")
    html.append("<table cellpadding=8>")
    html.append("<tr><th>name</th><th>input</th><th>pred</th><th>gold</th>"
                "<th>byte match</th></tr>")
    for stem, name, m in rows:
        html.append(
            f"<tr><td>{name}</td>"
            f"<td><img src='{stem}_input.png' width=256></td>"
            f"<td><img src='{stem}_pred.png' width=256></td>"
            f"<td><img src='{stem}_gold.png' width=256></td>"
            f"<td>{m}/1000</td></tr>")
    html.append("</table></body></html>")
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write("\n".join(html))

    volume.commit()
    avg = sum(m for _, _, m in rows) / max(1, len(rows))
    return {
        "n": len(rows),
        "avg_byte_match": round(avg, 1),
        "out_dir": OUT_DIR,
    }


@app.local_entrypoint()
def main(n: int = 12, seed: int = 0):
    print(sample.remote(n, seed))
