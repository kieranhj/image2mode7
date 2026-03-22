"""Download all gallery images from gallery_urls.txt with polite rate limiting."""
import time, urllib.request, pathlib, sys, os

URLS_FILE = 'gallery_urls.txt'
OUT_DIR   = pathlib.Path('test/horsenburger')
DELAY     = 0.2   # seconds between requests

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(URLS_FILE) as f:
    urls = [l.strip() for l in f if l.strip()]

total = len(urls)
skipped = downloaded = failed = 0

for i, url in enumerate(urls, 1):
    # Derive filename from the hash portion of the URL
    name = url.split('/media/')[1].replace('/', '_')
    dest = OUT_DIR / name

    if dest.exists():
        skipped += 1
        continue

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        dest.write_bytes(data)
        downloaded += 1
        if downloaded % 50 == 0 or i == total:
            print(f'[{i}/{total}] downloaded={downloaded} skipped={skipped} failed={failed}',
                  flush=True)
        time.sleep(DELAY)
    except Exception as e:
        failed += 1
        print(f'  FAIL {url}: {e}', file=sys.stderr, flush=True)

print(f'\nDone. downloaded={downloaded} skipped={skipped} failed={failed}')
