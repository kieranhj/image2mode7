"""Step 3 — prep_silver (parallel, preemption-resilient).

Stream a large image corpus, resize each to 256x192, run image2teletext.convert()
with 3 presets per image. Output is sharded webdataset tars under
/data/silver/silver-NNNN.tar (1000 source images = 3000 pairs per shard).

On restart, scans existing shards and skips that many images from the dataset
stream — preemption costs at most one in-flight shard's worth of work.

Target: 50k images x 3 presets = 150k pairs across 50 shards.
"""
import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_FILES = [
    "image2teletext.py",
    "teletext_decode.py",
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

OUT_DIR     = f"{DATA}/silver"
SHARD_FMT   = "silver-{:04d}.tar"
SHARD_GLOB  = "silver-*.tar"
SHARD_SIZE  = 1000        # source images per shard
PRESETS     = ["photo", "art", "vivid"]
DATASET     = "huggan/wikiart"
SPLIT       = "train"
IMG_COL     = "image"
N_WORKERS   = 16


def _worker_init():
    import os, sys
    os.cpu_count = lambda: 1
    sys.stderr = open(os.devnull, "w")
    sys.stdout = open(os.devnull, "w")


def _process_one(args):
    import sys, io
    sys.path.insert(0, "/repo")
    from PIL import Image
    from image2teletext import convert

    idx, jpeg_bytes = args
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    pages = []
    for preset in PRESETS:
        page = convert(img, preset=preset)
        assert len(page) == 1000
        pages.append(page)
    return idx, jpeg_bytes, pages


@app.function(volumes={DATA: volume}, cpu=N_WORKERS, timeout=12 * 60 * 60)
def prep_silver(n_images: int = 50_000):
    import sys, io, os, tarfile, time, glob
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, "/repo")
    from datasets import load_dataset
    from PIL import Image

    os.makedirs(OUT_DIR, exist_ok=True)
    volume.reload()

    existing = sorted(glob.glob(os.path.join(OUT_DIR, SHARD_GLOB)))
    completed_shards = len(existing)
    skip_n = completed_shards * SHARD_SIZE
    print(f"[prep_silver] {completed_shards} existing shards -> skipping "
          f"{skip_n} dataset rows", flush=True)
    print(f"[prep_silver] streaming {DATASET}[{SPLIT}], target={n_images} images, "
          f"presets={PRESETS}, workers={N_WORKERS}, shard_size={SHARD_SIZE}",
          flush=True)

    if completed_shards * SHARD_SIZE >= n_images:
        print("[prep_silver] target already reached, nothing to do.", flush=True)
        return {"already_done": True, "shards": completed_shards}

    ds = load_dataset(DATASET, split=SPLIT, streaming=True)
    if skip_n:
        ds = ds.skip(skip_n)

    def source():
        prepared = 0
        for row in ds:
            if skip_n + prepared >= n_images:
                return
            try:
                img = row[IMG_COL]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((256, 192), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=85)
                yield (skip_n + prepared, buf.getvalue())
                prepared += 1
            except Exception:
                continue

    t0 = time.time()
    written = skipped = 0
    in_shard = 0
    shard_idx = completed_shards
    shard_path = os.path.join(OUT_DIR, SHARD_FMT.format(shard_idx))
    shard_tmp  = shard_path + ".tmp"
    tar = tarfile.open(shard_tmp, "w")

    def close_shard(t, src, dst):
        t.close()
        os.replace(src, dst)
        volume.commit()

    try:
        with ProcessPoolExecutor(max_workers=N_WORKERS,
                                 initializer=_worker_init) as pool:
            for result in pool.map(_process_one, source(), chunksize=4):
                try:
                    idx, jpeg_bytes, pages = result
                except Exception:
                    skipped += 1
                    continue

                for preset, page in zip(PRESETS, pages):
                    key = f"{idx:07d}_{preset}"
                    ti = tarfile.TarInfo(f"{key}.jpg")
                    ti.size = len(jpeg_bytes)
                    tar.addfile(ti, io.BytesIO(jpeg_bytes))
                    ti = tarfile.TarInfo(f"{key}.bin")
                    ti.size = 1000
                    tar.addfile(ti, io.BytesIO(page))

                written += 1
                in_shard += 1

                if in_shard >= SHARD_SIZE:
                    close_shard(tar, shard_tmp, shard_path)
                    dt = time.time() - t0
                    rate = written / dt if dt else 0
                    total_done = (shard_idx + 1) * SHARD_SIZE
                    remaining  = n_images - total_done
                    eta = remaining / rate if rate else 0
                    print(f"  [shard {shard_idx} done -> {total_done}/{n_images}] "
                          f"skipped={skipped} rate={rate:.1f} img/s "
                          f"eta={eta/60:.1f} min", flush=True)
                    shard_idx += 1
                    in_shard = 0
                    shard_path = os.path.join(OUT_DIR, SHARD_FMT.format(shard_idx))
                    shard_tmp  = shard_path + ".tmp"
                    tar = tarfile.open(shard_tmp, "w")
    finally:
        # If the final shard has any rows but wasn't full, persist it anyway.
        tar.close()
        if in_shard > 0:
            os.replace(shard_tmp, shard_path)
            volume.commit()
        elif os.path.exists(shard_tmp):
            os.remove(shard_tmp)

    dt = time.time() - t0
    return {
        "dataset": DATASET,
        "written_this_run": written,
        "skipped_this_run": skipped,
        "shards_total": shard_idx + (1 if in_shard else 0),
        "out_dir": OUT_DIR,
        "wall_minutes": round(dt / 60, 1),
    }


@app.local_entrypoint()
def main(n_images: int = 50_000, spawn: bool = False):
    if spawn:
        call = prep_silver.spawn(n_images)
        print(f"spawned function call: {call.object_id}")
        print("track with: ./modal.bat app logs <app-id>")
    else:
        print(prep_silver.remote(n_images))
