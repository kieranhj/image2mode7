"""
Extract visible Mode 7 pages from dan.zip (teletextart.co.uk archive).

dan.zip contains .bin files of raw T42 packets:
    1092 bytes/page = 26 packets * 42 bytes (2-byte MRAG + 40 data)
Multi-subpage carousels are concatenations of 26-packet blocks.

We emit the 24 visible display rows (packets 1..24) per subpage with the
parity bit stripped, producing 960-byte Mode 7 pages ready for the
training pipeline.

Usage:
    python dan_to_pages.py [--zip PATH] [--out DIR]

Output:
    DIR/<edition>__<page>__sub<NN>.bin     960 bytes each
    DIR/manifest.json                      provenance per emitted page
"""

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

PAGE_BYTES = 1092
PACKET_BYTES = 42
ROW_DATA_BYTES = 40
VISIBLE_ROWS = 25  # rows 0..24 (row 0 = header)
VISIBLE_BYTES = VISIBLE_ROWS * ROW_DATA_BYTES  # 1000


def extract_subpage(packet_block: bytes) -> bytes:
    """Return 1000 bytes of visible display rows from one 1092-byte page block.

    Converts EBU broadcast control codes (0x00-0x1F) into BBC Micro Mode 7
    representation (0x80-0x9F) so the bytes match what teletext_decode.render_bytes
    and the rest of the pipeline expect.
    """
    out = bytearray(VISIBLE_BYTES)
    for row in range(VISIBLE_ROWS):
        pkt = packet_block[row * PACKET_BYTES:(row + 1) * PACKET_BYTES]
        data = bytearray(b & 0x7F for b in pkt[2:])
        for i, b in enumerate(data):
            if b < 0x20:
                data[i] = b | 0x80
        out[row * ROW_DATA_BYTES:(row + 1) * ROW_DATA_BYTES] = data
    return bytes(out)


def page_id_from_name(name: str) -> str:
    """Extract the page identifier (e.g. '100', '1a3') from a .bin filename."""
    stem = Path(name).stem.lower()
    m = re.match(r"([0-9a-f]{3})", stem)
    return m.group(1) if m else stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="datasets/teletextart/dan.zip", type=Path)
    ap.add_argument("--out", default="datasets/teletextart/pages", type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    skipped = 0

    with zipfile.ZipFile(args.zip) as zf:
        bin_infos = [i for i in zf.infolist()
                     if i.filename.lower().endswith(".bin")
                     and i.file_size > 0
                     and i.file_size % PAGE_BYTES == 0]

        for info in bin_infos:
            parts = info.filename.split("/")
            if len(parts) < 3:
                skipped += 1
                continue
            edition = parts[1]
            pass_dir = parts[-2]  # e.g. "binaries" or "binaries_v2"
            page_id = page_id_from_name(parts[-1])

            data = zf.read(info)
            n_subpages = len(data) // PAGE_BYTES

            for sub in range(n_subpages):
                block = data[sub * PAGE_BYTES:(sub + 1) * PAGE_BYTES]
                visible = extract_subpage(block)
                out_name = f"{edition}__{pass_dir}__{page_id}__sub{sub:02d}.bin"
                (args.out / out_name).write_bytes(visible)
                manifest.append({
                    "edition": edition,
                    "pass": pass_dir,
                    "page_id": page_id,
                    "subpage": sub,
                    "source": info.filename,
                    "out": out_name,
                    "sha256": hashlib.sha256(visible).hexdigest(),
                })

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} pages to {args.out}", flush=True)
    if skipped:
        print(f"Skipped {skipped} entries (no edition path)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
