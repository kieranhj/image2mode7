"""
make_gallery_html.py — Convert gallery images and produce a local browse page.

For each standard-aspect image in test/horsenburger/:
  - Runs image2teletext to get Mode 7 bytes
  - Renders the bytes back at original dimensions
  - Saves the render to test/horsenburger_conv/
  - Writes gallery.html with side-by-side pairs, match % and sort controls
"""

import pathlib, json, sys, argparse, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np
from PIL import Image

import image2teletext as m7
import teletext_decode as td

_ROOT        = pathlib.Path(__file__).parent.parent
GALLERY_DIR  = _ROOT / 'test/horsenburger'
CONV_DIR     = _ROOT / 'test/horsenburger_conv'
HTML_OUT     = pathlib.Path(__file__).parent / 'gallery.html'


def par_for_image(iw, ih):
    return (m7.MODE7_PIXEL_H / m7.MODE7_PIXEL_W) * (iw / ih)


def process_image(img_path):
    img_path = pathlib.Path(img_path)
    iw, ih = Image.open(img_path).size
    par = par_for_image(iw, ih)

    try:
        page = m7.convert_image(str(img_path),
                                direct_sample=True,
                                snap_palette=True, quant_colors=0,
                                smooth=0, par=par)
        rendered = td.render_bytes(page, iw, ih)

        # Save converted render
        conv_path = CONV_DIR / img_path.name
        rendered.save(conv_path)

        # Measure match
        cmap_orig, iw2, ih2 = td.load_and_quantise(img_path)
        rend_arr = np.array(rendered, dtype=np.float32)
        flat = rend_arr.reshape(-1, 3)
        dists = np.sum((flat[:, None, :] - td.PALETTE_RGB[None, :, :]) ** 2, axis=2)
        cmap_rend = np.argmin(dists, axis=1).reshape(ih2, iw2).astype(np.uint8)
        sp_orig = td.sample_subpixels(cmap_orig, iw2, ih2)
        sp_rend = td.sample_subpixels(cmap_rend, iw2, ih2)

        matching = sum(
            1 for row in range(td.N_ROWS) for col in range(1, td.N_COLS)  # skip col 0
            if np.array_equal(
                sp_orig[row*3:(row+1)*3, col*2:(col+1)*2],
                sp_rend[row*3:(row+1)*3, col*2:(col+1)*2]
            )
        )
        total = td.N_ROWS * (td.N_COLS - 1)  # 25 × 39 = 975
        match_pct = 100.0 * matching / total

        return {'file': img_path.name, 'match_pct': round(match_pct, 1),
                'size': f'{iw}x{ih}', 'ok': True}

    except Exception as e:
        return {'file': img_path.name, 'match_pct': 0.0,
                'size': f'{iw}x{ih}', 'ok': False, 'error': str(e)}


HTML_TEMPLATE = '''\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Teletext Gallery — Converter Comparison</title>
<style>
  body {{ font-family: monospace; background: #111; color: #ccc; margin: 0; padding: 8px; }}
  h1 {{ color: #fff; font-size: 1.1em; margin: 4px 0 8px; }}
  #controls {{ margin-bottom: 8px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  #controls label {{ color: #aaa; font-size: 0.85em; }}
  #controls input, #controls select {{ background: #222; color: #ccc; border: 1px solid #444;
    padding: 3px 6px; border-radius: 3px; font-family: monospace; }}
  #stats {{ font-size: 0.8em; color: #888; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 8px; }}
  .card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 4px;
           padding: 6px; display: flex; flex-direction: column; gap: 4px; }}
  .card.hidden {{ display: none; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
  .pair img {{ width: 100%; height: auto; display: block; image-rendering: pixelated; }}
  .label {{ font-size: 0.7em; color: #666; text-align: center; }}
  .meta {{ font-size: 0.78em; display: flex; justify-content: space-between; align-items: center; }}
  .pct {{ font-weight: bold; }}
  .pct.good  {{ color: #4c4; }}
  .pct.ok    {{ color: #cc4; }}
  .pct.poor  {{ color: #c44; }}
  .filename  {{ color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                max-width: 55%; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Teletext Gallery — Original vs Converter (cols 1–39, skipping col 0)</h1>
<div id="controls">
  <label>Sort:
    <select id="sortSel" onchange="applyFilters()">
      <option value="pct-asc">Match % ↑ (worst first)</option>
      <option value="pct-desc">Match % ↓ (best first)</option>
      <option value="name">Filename</option>
    </select>
  </label>
  <label>Min match:
    <input type="range" id="minPct" min="0" max="100" value="0" step="5"
           oninput="document.getElementById('minVal').textContent=this.value+'%'; applyFilters()">
    <span id="minVal">0%</span>
  </label>
  <label>Max match:
    <input type="range" id="maxPct" min="0" max="100" value="100" step="5"
           oninput="document.getElementById('maxVal').textContent=this.value+'%'; applyFilters()">
    <span id="maxVal">100%</span>
  </label>
  <span id="stats"></span>
</div>
<div class="grid" id="grid"></div>

<script>
const data = {DATA};

function pctClass(p) {{
  return p >= 80 ? 'good' : p >= 50 ? 'ok' : 'poor';
}}

function buildCards() {{
  const grid = document.getElementById('grid');
  data.forEach(d => {{
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.pct  = d.pct;
    card.dataset.name = d.file;
    card.innerHTML = `
      <div class="meta">
        <span class="filename" title="${{d.file}}">${{d.file}}</span>
        <span class="pct ${{pctClass(d.pct)}}">${{d.pct.toFixed(1)}}%</span>
        <span style="color:#555">${{d.size}}</span>
      </div>
      <div class="pair">
        <div>
          <img src="../test/horsenburger/${{d.file}}" loading="lazy">
          <div class="label">original</div>
        </div>
        <div>
          <img src="../test/horsenburger_conv/${{d.file}}" loading="lazy">
          <div class="label">converted</div>
        </div>
      </div>`;
    grid.appendChild(card);
  }});
}}

function applyFilters() {{
  const minP = +document.getElementById('minPct').value;
  const maxP = +document.getElementById('maxPct').value;
  const sort = document.getElementById('sortSel').value;
  const cards = Array.from(document.querySelectorAll('.card'));

  cards.sort((a, b) => {{
    if (sort === 'pct-asc')  return +a.dataset.pct - +b.dataset.pct;
    if (sort === 'pct-desc') return +b.dataset.pct - +a.dataset.pct;
    return a.dataset.name.localeCompare(b.dataset.name);
  }});

  const grid = document.getElementById('grid');
  let visible = 0;
  cards.forEach(c => {{
    const p = +c.dataset.pct;
    const show = p >= minP && p <= maxP;
    c.classList.toggle('hidden', !show);
    if (show) visible++;
    grid.appendChild(c);
  }});
  document.getElementById('stats').textContent =
    `Showing ${{visible}} / ${{cards.length}} images`;
}}

buildCards();
applyFilters();
</script>
</body>
</html>
'''


def write_html(results):
    data = json.dumps([
        {'file': r['file'], 'pct': r['match_pct'], 'size': r.get('size', '?')}
        for r in results
    ])
    html = HTML_TEMPLATE.replace('{DATA}', data)
    HTML_OUT.write_text(html, encoding='utf-8')
    print(f'Written {HTML_OUT}  ({len(results)} images)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0,
                        help='Process only first N images (0 = all)')
    parser.add_argument('--reuse-json', metavar='FILE',
                        help='Skip conversion, load results from existing JSON and just rebuild HTML')
    args = parser.parse_args()

    CONV_DIR.mkdir(parents=True, exist_ok=True)

    if args.reuse_json:
        results = json.loads(pathlib.Path(args.reuse_json).read_text())
        write_html(results)
        return

    imgs = sorted(GALLERY_DIR.glob('*.png'))
    imgs = [p for p in imgs
            if 0.7 <= Image.open(p).size[0] / Image.open(p).size[1] <= 1.6]
    if args.limit:
        imgs = imgs[:args.limit]

    print(f'Processing {len(imgs)} images...', flush=True)
    results = []
    for i, img in enumerate(imgs, 1):
        r = process_image(img)
        results.append(r)
        status = f"{r['match_pct']:.1f}%" if r['ok'] else f"ERROR: {r.get('error','')}"
        if i % 50 == 0 or i == len(imgs):
            print(f'[{i}/{len(imgs)}] {img.name}: {status}', flush=True)

    # Save JSON alongside HTML for reuse
    (pathlib.Path(__file__).parent / 'gallery_results_full.json').write_text(json.dumps(results, indent=2))
    write_html(results)

    valid = [r for r in results if r['ok']]
    if valid:
        import statistics
        pcts = [r['match_pct'] for r in valid]
        print(f'\nMean: {statistics.mean(pcts):.1f}%  '
              f'Median: {statistics.median(pcts):.1f}%  '
              f'>=80%: {sum(1 for p in pcts if p>=80)}  '
              f'<50%: {sum(1 for p in pcts if p<50)}')


if __name__ == '__main__':
    main()
