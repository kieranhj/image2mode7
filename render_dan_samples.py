"""Render a handful of extracted dan.zip pages as PNGs for visual sanity check."""
import sys
from pathlib import Path
from teletext_decode import render_bytes

SAMPLES = [
    "bbc1_1991-10-31_Dan__binaries__100__sub00.bin",
    "bbc1_1991-10-31_Dan__binaries__101__sub00.bin",
    "bbc1_1991-10-31_Dan__binaries__120__sub00.bin",
    "gmtv_1993-02-22_Dan__binaries__100__sub00.bin",
    "sky-one_1993-02-16_Dan__binaries__100__sub00.bin",
    "tvam_1992-10-20_Dan__binaries__100__sub00.bin",
]

src = Path("datasets/teletextart/pages")
dst = Path("datasets/teletextart/sample_renders")
dst.mkdir(parents=True, exist_ok=True)

for name in SAMPLES:
    p = src / name
    if not p.exists():
        print(f"missing: {name}")
        continue
    img = render_bytes(p.read_bytes())
    out = dst / (p.stem + ".png")
    img.save(out)
    print(f"rendered {out}")
