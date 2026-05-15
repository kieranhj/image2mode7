"""
Scrape teletext-tagged artworks from 16colo.rs.

Uses the per-pack zip download (one request per pack instead of per file)
to minimise load. Honours robots.txt Crawl-delay: 10.

Usage:
    python scrape_16colors_teletext.py [--out DIR] [--delay SECONDS]

Output:
    DIR/images/<PACK>__<FILENAME>            extracted teletext images
    DIR/zips/<PACK>.zip                      cached pack archives (skipped on rerun)
    DIR/manifest.json                        {pack_id, filename, year, bytes, sha256}
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

BASE = "https://16colo.rs"
TAG_URL = f"{BASE}/tags/content/teletext"
UA = "image2mode7-research/0.1 (khconnell@gmail.com)"

IMAGE_EXTS = (".PNG", ".GIF", ".JPG", ".JPEG", ".BMP")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parse_listing(html: str) -> dict[str, set[str]]:
    """Return {pack_id: {filename, ...}} for image entries on the tag page."""
    pattern = re.compile(
        r"/pack/([a-z0-9]+)/([A-Z0-9_'.\-]+\.(?:PNG|GIF|JPG|JPEG|BMP))"
    )
    packs: dict[str, set[str]] = {}
    for pack_id, fname in pattern.findall(html):
        packs.setdefault(pack_id, set()).add(fname)
    return packs


def find_zip_url(pack_html: str, pack_id: str) -> str | None:
    """Locate the /archive/<year>/<pack_id>.zip link on the pack page."""
    m = re.search(
        rf'/archive/(\d{{4}})/{re.escape(pack_id)}\.zip', pack_html, re.IGNORECASE
    )
    if not m:
        return None
    return f"{BASE}/archive/{m.group(1)}/{pack_id}.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/16colors_teletext", type=Path)
    ap.add_argument("--delay", default=10.0, type=float,
                    help="Seconds between HTTP requests (robots.txt: 10)")
    args = ap.parse_args()

    images_dir = args.out / "images"
    zips_dir = args.out / "zips"
    images_dir.mkdir(parents=True, exist_ok=True)
    zips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"

    def polite_get(url: str) -> bytes:
        if polite_get.last_t is not None:
            wait = args.delay - (time.monotonic() - polite_get.last_t)
            if wait > 0:
                time.sleep(wait)
        try:
            data = fetch(url)
        finally:
            polite_get.last_t = time.monotonic()
        return data
    polite_get.last_t = None

    print(f"GET {TAG_URL}", flush=True)
    listing_html = polite_get(TAG_URL).decode("utf-8", errors="replace")
    packs = parse_listing(listing_html)
    total_files = sum(len(v) for v in packs.values())
    print(f"Found {total_files} teletext images across {len(packs)} packs",
          flush=True)

    manifest: list[dict] = []

    for i, (pack_id, wanted) in enumerate(sorted(packs.items()), 1):
        print(f"[{i}/{len(packs)}] pack {pack_id} ({len(wanted)} files)",
              flush=True)
        zip_path = zips_dir / f"{pack_id}.zip"
        year = None

        if not zip_path.exists():
            try:
                pack_html = polite_get(f"{BASE}/pack/{pack_id}/").decode(
                    "utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                print(f"  pack page {pack_id}: HTTP {e.code}, skipping",
                      flush=True)
                continue
            zip_url = find_zip_url(pack_html, pack_id)
            if not zip_url:
                print(f"  no zip URL found for {pack_id}, skipping", flush=True)
                continue
            year = zip_url.split("/archive/")[1].split("/")[0]
            print(f"  GET {zip_url}", flush=True)
            try:
                zip_bytes = polite_get(zip_url)
            except urllib.error.HTTPError as e:
                print(f"  zip {pack_id}: HTTP {e.code}, skipping", flush=True)
                continue
            zip_path.write_bytes(zip_bytes)
        else:
            print(f"  cached: {zip_path.name}", flush=True)

        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = {n.upper(): n for n in zf.namelist()}
                for fname in sorted(wanted):
                    real = names.get(fname.upper())
                    if real is None:
                        print(f"    missing in zip: {fname}", flush=True)
                        continue
                    data = zf.read(real)
                    out_name = f"{pack_id}__{fname}"
                    (images_dir / out_name).write_bytes(data)
                    manifest.append({
                        "pack_id": pack_id,
                        "filename": fname,
                        "year": year,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "out": out_name,
                    })
        except zipfile.BadZipFile:
            print(f"  bad zip: {zip_path}", flush=True)
            continue

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} images to {images_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
