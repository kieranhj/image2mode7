"""Step 2 — prep_gold.

Download the ~1900 horsenburger gallery PNGs to the persistent volume, encode
each one to its 1000-byte Mode 7 stream via image2teletext.convert (DP solver),
and write `/data/gold.npz` containing:
  bytes : (N, 1000) uint8
  names : (N,) <U... filenames

Re-runnable: the downloader skips existing files, and convert is deterministic.
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
N_WORKERS   = 16


def _worker_init():
    import os, sys
    os.cpu_count = lambda: 1
    sys.stderr = open(os.devnull, "w")
    sys.stdout = open(os.devnull, "w")


def _process_one(path_str):
    import sys
    sys.path.insert(0, "/repo")
    from PIL import Image
    from image2teletext import convert
    img = Image.open(path_str).convert("RGB").resize((256, 192), Image.LANCZOS)
    page = convert(img)
    assert len(page) == 1000
    return path_str, bytes(page)


@app.function(volumes={DATA: volume}, cpu=N_WORKERS, timeout=2 * 60 * 60)
def prep_gold():
    import os, sys, runpy, pathlib, numpy as np, time
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    sys.path.insert(0, "/repo")
    os.environ["GALLERY_OUT_DIR"] = GALLERY_DIR
    pathlib.Path(GALLERY_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[prep_gold] downloading gallery to {GALLERY_DIR}", flush=True)
    runpy.run_path("/repo/gallery/download_gallery.py", run_name="__main__")
    volume.commit()

    pngs = sorted(pathlib.Path(GALLERY_DIR).glob("*.png")) \
         + sorted(pathlib.Path(GALLERY_DIR).glob("*.PNG"))
    paths = [str(p) for p in pngs]
    print(f"[prep_gold] encoding {len(paths)} PNGs via DP solver "
          f"({N_WORKERS} workers) -> {OUT_NPZ}", flush=True)

    t0 = time.time()
    name_to_bytes: dict[str, bytes] = {}
    failed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_worker_init) as pool:
        for result in tqdm(pool.map(_process_one, paths, chunksize=4),
                           total=len(paths)):
            try:
                path_str, page = result
                name_to_bytes[pathlib.Path(path_str).name] = page
            except Exception as e:
                failed += 1
                print(f"  FAIL: {e}", flush=True)

    names = [p.name for p in pngs if p.name in name_to_bytes]
    rows = [np.frombuffer(name_to_bytes[n], dtype=np.uint8) for n in names]
    arr_bytes = np.stack(rows, axis=0) if rows else np.zeros((0, 1000), np.uint8)
    arr_names = np.array(names)
    np.savez(OUT_NPZ, bytes=arr_bytes, names=arr_names)
    volume.commit()

    return {
        "gallery_pngs": len(pngs),
        "decoded": int(arr_bytes.shape[0]),
        "failed": failed,
        "wall_minutes": round((time.time() - t0) / 60, 1),
        "npz_path": OUT_NPZ,
        "npz_shape": list(arr_bytes.shape),
    }


@app.local_entrypoint()
def main():
    print(prep_gold.remote())
