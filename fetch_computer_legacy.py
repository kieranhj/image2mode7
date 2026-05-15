"""
Download high-quality teletext recoveries from computer-legacy.com.

Catalogue (JSON) is fetched once from /res/teletext/data.php; files are then
pulled from /res/teletext/{tape_id:04d}/{filename}.

Default selection: every recovery with a finalised (hand-edited) version,
plus all 4-star and 5-star squashed recoveries.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

CATALOGUE_URL = "https://computer-legacy.com/res/teletext/data.php"
FILE_URL_TMPL = "https://computer-legacy.com/res/teletext/{tape:04d}/{name}"
UA = "image2mode7-research/0.1 (khconnell@gmail.com)"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def select(recs: list[dict], min_rating: int, want_finalised: bool):
    """Return (file_kind, record) tuples for every download we want."""
    jobs: list[tuple[str, str, dict]] = []
    for r in recs:
        if want_finalised and r.get("link_finalised"):
            jobs.append(("finalised", r["link_finalised"], r))
        rating = r.get("rating") or 0
        if rating >= min_rating and r.get("link_squashed"):
            jobs.append(("squashed", r["link_squashed"], r))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/computer_legacy", type=Path)
    ap.add_argument("--delay", default=1.0, type=float,
                    help="Seconds between requests")
    ap.add_argument("--min-rating", default=4, type=int,
                    help="Minimum star rating for squashed downloads (default 4)")
    ap.add_argument("--no-finalised", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    zips_dir = args.out / "zips"
    zips_dir.mkdir(exist_ok=True)

    print(f"GET {CATALOGUE_URL}", flush=True)
    cat_bytes = fetch(CATALOGUE_URL)
    (args.out / "recoveries.json").write_bytes(cat_bytes)
    recs = json.loads(cat_bytes)
    print(f"catalogue: {len(recs)} recoveries", flush=True)

    jobs = select(recs, args.min_rating, want_finalised=not args.no_finalised)
    by_kind: dict[str, int] = {}
    for kind, _, _ in jobs:
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(f"selected {len(jobs)} files: {by_kind}", flush=True)

    manifest: list[dict] = []
    last_t = 0.0
    for i, (kind, name, rec) in enumerate(jobs, 1):
        url = FILE_URL_TMPL.format(tape=rec["tape_id"], name=name)
        out_name = f"tape{rec['tape_id']:04d}__{kind}__{name}"
        out_path = zips_dir / out_name
        if out_path.exists():
            print(f"[{i}/{len(jobs)}] cached: {out_name}", flush=True)
        else:
            wait = args.delay - (time.monotonic() - last_t)
            if wait > 0:
                time.sleep(wait)
            print(f"[{i}/{len(jobs)}] GET {url}", flush=True)
            try:
                data = fetch(url)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                last_t = time.monotonic()
                continue
            out_path.write_bytes(data)
            last_t = time.monotonic()

        size = out_path.stat().st_size
        manifest.append({
            "kind": kind,
            "filename": name,
            "out": out_name,
            "tape_id": rec["tape_id"],
            "recovery_date": rec["recovery_date"],
            "channel": rec["channel"],
            "programme": rec["programme"],
            "rating": rec["rating"],
            "rating_notes": rec.get("rating_notes"),
            "bytes": size,
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        })

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_mb = sum(m["bytes"] for m in manifest) / 1e6
    print(f"\ndownloaded {len(manifest)} files, {total_mb:.1f} MB total",
          flush=True)
    print(f"manifest: {args.out / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
