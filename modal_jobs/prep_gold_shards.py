"""Step M3.1 — prep_gold_shards.

Pair the 1909 horsenburger PNGs with their DP-solver bytes from gold.npz and
write a webdataset shard at /data/gold/gold-0000.tar (jpg @ 256x192 + bin
pairs), same format as silver / bronze, so train.py can mix them in.
"""
import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pillow==11.0.0", "numpy==2.1.3",
)

volume = modal.Volume.from_name("teletext-m1", create_if_missing=True)
app    = modal.App("teletext-m1-gold-shards", image=image)
DATA   = "/data"

GALLERY_DIR = f"{DATA}/horsenburger"
GOLD_NPZ    = f"{DATA}/gold.npz"
OUT_DIR     = f"{DATA}/gold"
SHARD_PATH  = f"{OUT_DIR}/gold-0000.tar"


@app.function(volumes={DATA: volume}, timeout=30 * 60)
def prep_gold_shards():
    import os, io, tarfile, numpy as np
    from PIL import Image

    volume.reload()
    os.makedirs(OUT_DIR, exist_ok=True)

    gold = np.load(GOLD_NPZ)
    names = list(gold["names"])
    bytes_arr = gold["bytes"]
    print(f"[prep_gold_shards] {len(names)} pairs from {GOLD_NPZ}", flush=True)

    tmp = SHARD_PATH + ".tmp"
    written = skipped = 0
    with tarfile.open(tmp, "w") as tar:
        for i, name in enumerate(names):
            png_path = os.path.join(GALLERY_DIR, str(name))
            if not os.path.exists(png_path):
                skipped += 1
                continue
            img = Image.open(png_path).convert("RGB").resize(
                (256, 192), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            jpeg_bytes = buf.getvalue()
            page = bytes(bytes_arr[i].tobytes())

            key = f"{i:07d}"
            ti = tarfile.TarInfo(f"{key}.jpg")
            ti.size = len(jpeg_bytes)
            tar.addfile(ti, io.BytesIO(jpeg_bytes))
            ti = tarfile.TarInfo(f"{key}.bin")
            ti.size = 1000
            tar.addfile(ti, io.BytesIO(page))
            written += 1

    os.replace(tmp, SHARD_PATH)
    volume.commit()
    return {
        "shard": SHARD_PATH,
        "written": written,
        "skipped": skipped,
    }


@app.local_entrypoint()
def main():
    print(prep_gold_shards.remote())
