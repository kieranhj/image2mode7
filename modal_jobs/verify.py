"""Smoke test: build the image, mount the volume, import the repo."""
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


@app.function(volumes={DATA: volume})
def verify():
    import sys, pathlib, torch, transformers
    sys.path.insert(0, "/repo")
    import image2teletext, teletext_decode  # noqa: F401

    marker = pathlib.Path(f"{DATA}/verify.txt")
    marker.write_text("ok")
    volume.commit()
    return {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "repo_imports_ok": True,
        "volume_round_trip": marker.read_text(),
    }


@app.local_entrypoint()
def main():
    print(verify.remote())
