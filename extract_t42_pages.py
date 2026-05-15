"""
Extract 1000-byte BBC Mode 7 pages from raw T42 packet streams.

Walks one or more directories of .zip files (each containing a single .t42
stream from a broadcast capture), uses the teletext library to paginate the
stream, and writes each subpage as a 1000-byte file in BBC Mode 7 format.

Each emitted page is 25 rows × 40 bytes:
    - parity bit stripped from every byte (& 0x7F)
    - EBU broadcast control codes (0x00-0x1F) shifted to BBC Mode 7 form
      (0x80-0x9F) so the bytes round-trip through teletext_decode.render_bytes
    - missing packets filled with spaces (0x20)

Usage:
    python extract_t42_pages.py --in-dir DIR [--in-dir DIR ...] --out DIR
"""

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from teletext.file import FileChunker
from teletext.packet import Packet
from teletext.service import Service

ROW_BYTES = 40
N_ROWS = 25
PAGE_BYTES = N_ROWS * ROW_BYTES  # 1000


def to_bbc_row(seven_bit: bytes) -> bytes:
    """Convert one 40-byte EBU row into BBC Mode 7 form (controls 0x80-0x9F)."""
    out = bytearray(seven_bit[:ROW_BYTES])
    for i, b in enumerate(out):
        if b < 0x20:
            out[i] = b | 0x80
    if len(out) < ROW_BYTES:
        out.extend(b" " * (ROW_BYTES - len(out)))
    return bytes(out)


def page_from_subpage(sp) -> bytes:
    out = bytearray(PAGE_BYTES)
    for row in range(N_ROWS):
        if sp.has_packet(row):
            data = sp.packet(row).sevenbit
        else:
            data = b" " * ROW_BYTES
        out[row * ROW_BYTES:(row + 1) * ROW_BYTES] = to_bbc_row(data)
    return bytes(out)


def iter_t42_streams(zip_path: Path):
    """Yield (member_name, raw_t42_bytes) for each .t42 in a zip."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".t42"):
                yield name, zf.read(name)


def extract_zip(zip_path: Path, out_dir: Path, prefix: str) -> list[dict]:
    """Extract all pages from one zip; return manifest entries."""
    entries: list[dict] = []
    for member, data in iter_t42_streams(zip_path):
        n_packets = len(data) // 42
        if n_packets == 0:
            continue
        packets = (Packet(d, n) for n, d in FileChunker(io.BytesIO(data), 42))
        try:
            svc = Service.from_packets(packets)
            subpages = list(svc.all_subpages)
        except Exception as e:
            print(f"  parse failed for {member}: {e}", flush=True)
            continue

        for idx, sp in enumerate(subpages):
            try:
                page_id = str(sp.mrg_PN).lower()
                sub_id = sp.mrg_SC if isinstance(sp.mrg_SC, int) else 0
            except Exception:
                page_id = f"unk{idx:04d}"
                sub_id = 0
            page_id = re.sub(r"[^0-9a-fA-F]", "_", page_id)[:4] or "x"
            page_bytes = page_from_subpage(sp)
            out_name = f"{prefix}__{page_id}__sub{sub_id:02x}__{idx:05d}.bin"
            (out_dir / out_name).write_bytes(page_bytes)
            entries.append({
                "source_zip": zip_path.name,
                "stream_member": member,
                "page_id": page_id,
                "subpage": sub_id,
                "stream_index": idx,
                "out": out_name,
                "sha256": hashlib.sha256(page_bytes).hexdigest(),
            })
        print(f"  {member}: {n_packets} packets -> {len(subpages)} pages",
              flush=True)
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", action="append", required=True, type=Path,
                    help="Directory of zips containing .t42 streams. Repeatable.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for extracted pages")
    ap.add_argument("--prefix-from-stem", action="store_true",
                    help="Use the zip filename stem as the per-page prefix")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for in_dir in args.in_dir:
        zips = sorted(in_dir.glob("*.zip"))
        print(f"--- {in_dir}: {len(zips)} zips ---", flush=True)
        for i, zp in enumerate(zips, 1):
            prefix = zp.stem if args.prefix_from_stem else zp.parent.name + "_" + zp.stem
            print(f"[{i}/{len(zips)}] {zp.name}", flush=True)
            entries = extract_zip(zp, args.out, prefix=prefix)
            for e in entries:
                e["source_dir"] = str(in_dir)
            manifest.extend(entries)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} pages to {args.out}", flush=True)
    print(f"Manifest: {args.out / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
