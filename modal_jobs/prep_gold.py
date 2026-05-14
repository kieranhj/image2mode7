"""Step 2 — prep_gold.

Download the ~1900 horsenburger gallery PNGs to the persistent volume, decode
each one to its 1000-byte Mode 7 stream, and write `/data/gold.npz` containing:
  bytes : (N, 1000) uint8
  names : (N,) <U... filenames

Re-runnable: the downloader skips existing files, and decode is idempotent.
"""
import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_FILES = [
    "image2teletext.py",
    "teletext_decode.py",
    "gallery/download_gallery.py",
    "gallery/gallery_urls.txt",
]

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "torchvision==0.20.1",
    "transformers==4.46.3", "accelerate==1.1.1",
    "datasets==3.1.0", "webdataset==0.2.100",
    "pillow==11.0.0", "numpy==2.1.3", "tqdm",
)
for _f in REPO_FILES:
    image = image.add_local_file(str(REPO_ROOT / _f), f"/repo/{_f}")

volume = modal.Volume.from_name("teletext-m1", create_if_missing=True)
app    = modal.App("teletext-m1", image=image)
DATA   = "/data"

GALLERY_DIR = f"{DATA}/horsenburger"
OUT_NPZ     = f"{DATA}/gold.npz"


@app.function(volumes={DATA: volume}, timeout=60 * 60)
def prep_gold():
    import os, sys, runpy, pathlib, numpy as np
    from tqdm import tqdm

    sys.path.insert(0, "/repo")
    os.environ["GALLERY_OUT_DIR"] = GALLERY_DIR
    pathlib.Path(GALLERY_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[prep_gold] downloading gallery to {GALLERY_DIR}", flush=True)
    runpy.run_path("/repo/gallery/download_gallery.py", run_name="__main__")
    volume.commit()

    import teletext_decode

    pngs = sorted(pathlib.Path(GALLERY_DIR).glob("*.png")) \
         + sorted(pathlib.Path(GALLERY_DIR).glob("*.PNG"))
    print(f"[prep_gold] decoding {len(pngs)} PNGs -> {OUT_NPZ}", flush=True)

    names, rows, failed = [], [], 0
    for p in tqdm(pngs):
        try:
            page = teletext_decode.decode_image(str(p))
            assert len(page) == 1000
            rows.append(np.frombuffer(bytes(page), dtype=np.uint8))
            names.append(p.name)
        except Exception as e:
            failed += 1
            print(f"  FAIL {p.name}: {e}", flush=True)

    arr_bytes = np.stack(rows, axis=0) if rows else np.zeros((0, 1000), np.uint8)
    arr_names = np.array(names)
    np.savez(OUT_NPZ, bytes=arr_bytes, names=arr_names)
    volume.commit()

    return {
        "gallery_pngs": len(pngs),
        "decoded": int(arr_bytes.shape[0]),
        "failed": failed,
        "npz_path": OUT_NPZ,
        "npz_shape": list(arr_bytes.shape),
    }


@app.local_entrypoint()
def main():
    print(prep_gold.remote())
