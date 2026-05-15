"""Step M2.1 — prep_bronze.

Take the uploaded /data/bronze.tar.gz (containing ~112k 1000-byte .bin teletext
streams), extract, and for each stream:
  - render via teletext_decode.render_bytes -> RGB image
  - resize to 256x192
  - encode as JPEG
  - write (jpg, bin) pair to bronze-NNNN.tar webdataset shards (1000 per shard)

Same shard format as silver, so train.py can mix them with wds.RandomMix.
Resumable: glob existing shards, skip that many .bin files.
"""
import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_FILES = ["teletext_decode.py"]

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pillow==11.0.0", "numpy==2.1.3", "tqdm",
)
for _f in REPO_FILES:
    image = image.add_local_file(str(REPO_ROOT / _f), f"/repo/{_f}")

volume = modal.Volume.from_name("teletext-m1", create_if_missing=True)
app    = modal.App("teletext-m1-bronze", image=image)
DATA   = "/data"

SRC_TGZ     = f"{DATA}/bronze.tar.gz"
EXTRACT_DIR = f"{DATA}/bronze_pages"
OUT_DIR     = f"{DATA}/bronze"
SHARD_FMT   = "bronze-{:04d}.tar"
SHARD_GLOB  = "bronze-*.tar"
SHARD_SIZE  = 1000
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
    import teletext_decode as td

    idx, bin_path = args
    with open(bin_path, "rb") as f:
        page = f.read()
    if len(page) != 1000:
        return None
    img = td.render_bytes(page).convert("RGB").resize(
        (256, 192), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return idx, buf.getvalue(), page


@app.function(volumes={DATA: volume}, cpu=N_WORKERS, timeout=4 * 60 * 60)
def prep_bronze():
    import sys, io, os, tarfile, glob, time
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, "/repo")

    os.makedirs(OUT_DIR, exist_ok=True)
    volume.reload()

    # Extract source archive once.
    if not os.path.isdir(EXTRACT_DIR) or not os.listdir(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        print(f"[prep_bronze] extracting {SRC_TGZ} -> {EXTRACT_DIR}", flush=True)
        with tarfile.open(SRC_TGZ, "r:gz") as t:
            t.extractall(EXTRACT_DIR)
        volume.commit()
    else:
        print(f"[prep_bronze] reusing existing {EXTRACT_DIR}", flush=True)

    bins = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**", "*.bin"),
                            recursive=True))
    print(f"[prep_bronze] found {len(bins)} .bin files", flush=True)

    existing = sorted(glob.glob(os.path.join(OUT_DIR, SHARD_GLOB)))
    completed_shards = len(existing)
    skip_n = completed_shards * SHARD_SIZE
    print(f"[prep_bronze] {completed_shards} existing shards -> skipping "
          f"{skip_n} sources", flush=True)

    work = [(skip_n + i, p) for i, p in enumerate(bins[skip_n:])]
    if not work:
        return {"already_done": True, "shards": completed_shards}

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
            for result in pool.map(_process_one, work, chunksize=8):
                if result is None:
                    skipped += 1
                    continue
                idx, jpeg_bytes, page = result
                key = f"{idx:07d}"
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
                    print(f"  [shard {shard_idx} done] "
                          f"written={written} skipped={skipped} "
                          f"rate={rate:.1f} img/s", flush=True)
                    shard_idx += 1
                    in_shard = 0
                    shard_path = os.path.join(OUT_DIR, SHARD_FMT.format(shard_idx))
                    shard_tmp  = shard_path + ".tmp"
                    tar = tarfile.open(shard_tmp, "w")
    finally:
        tar.close()
        if in_shard > 0:
            os.replace(shard_tmp, shard_path)
            volume.commit()
        elif os.path.exists(shard_tmp):
            os.remove(shard_tmp)

    dt = time.time() - t0
    return {
        "bins_total": len(bins),
        "written_this_run": written,
        "skipped_this_run": skipped,
        "shards_total": shard_idx + (1 if in_shard else 0),
        "out_dir": OUT_DIR,
        "wall_minutes": round(dt / 60, 1),
    }


@app.local_entrypoint()
def main(spawn: bool = False):
    if spawn:
        call = prep_bronze.spawn()
        print(f"spawned: {call.object_id}")
    else:
        print(prep_bronze.remote())
