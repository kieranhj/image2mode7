"""
Decode all 16colo.rs teletext-tagged images to 1000-byte BBC Mode 7 pages
via teletext_decode.decode_image.

Each image is the artist's rendered output (PNG/JPG/GIF). The decoder samples
the 80x75 sub-pixel grid and reconstructs a byte stream that re-renders to
approximately the same image. Lossy step (graphics-only decoder, JPEG noise),
but the resulting bytes are still teletext-syntactically valid.

Usage:
    python decode_16colors_to_bytes.py
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from teletext_decode import decode_image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",
                    default="datasets/16colors_teletext/images", type=Path)
    ap.add_argument("--out",
                    default="datasets/16colors_teletext/pages", type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted(args.in_dir.iterdir())
    print(f"decoding {len(images)} images", flush=True)

    manifest: list[dict] = []
    failures: list[dict] = []
    for i, p in enumerate(images, 1):
        out_name = p.stem + ".bin"
        out_path = args.out / out_name
        if out_path.exists():
            data = out_path.read_bytes()
        else:
            try:
                data = bytes(decode_image(str(p)))
            except Exception as e:
                failures.append({"src": p.name, "error": str(e)})
                continue
            out_path.write_bytes(data)
        manifest.append({
            "src": p.name,
            "out": out_name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        })
        if i % 200 == 0:
            print(f"  [{i}/{len(images)}]", flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if failures:
        (args.out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"\ndecoded {len(manifest)} pages", flush=True)
    if failures:
        print(f"failures: {len(failures)} (see failures.json)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
