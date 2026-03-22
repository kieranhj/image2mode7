"""
test_gallery.py — Run image2teletext.py on the horsenburger gallery images and
measure how closely the rendered output matches the original.

Quality metric: for each of the 1000 character cells, sample the 6 sub-pixels
from both the original (after quantising to 8 colours) and the rendered output.
A cell "matches" if all 6 sub-pixels agree.

Usage:
    python test_gallery.py [--limit N] [--workers N] [--out results.json]
"""

import pathlib, json, sys, argparse, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

import image2teletext as m7
import teletext_decode as td

GALLERY_DIR = pathlib.Path(__file__).parent.parent / 'test/horsenburger'

# Derive the PAR that forces exactly 39 solver columns (frame_w=39, pw=78)
# for an image of given dimensions, regardless of its aspect ratio.
# Formula: par = (MODE7_PIXEL_H / MODE7_PIXEL_W) * (img_w / img_h)
#                = (75 / 78) * (img_w / img_h)
def par_for_image(img_w, img_h):
    return (m7.MODE7_PIXEL_H / m7.MODE7_PIXEL_W) * (img_w / img_h)


def settings_for_image(img_path):
    img = Image.open(img_path)
    iw, ih = img.size
    return dict(
        direct_sample=True,   # point-sample at exact SP grid — avoids bilinear blur
        snap_palette=True,
        quant_colors=0,
        smooth=0,
        par=par_for_image(iw, ih),
    )


def test_one(img_path, settings=None):
    """
    Convert img_path with image2teletext, render back, compare with original.
    Returns a dict with match stats and error info.
    """
    img_path = pathlib.Path(img_path)
    if settings is None:
        settings = settings_for_image(img_path)

    try:
        t0 = time.time()
        page = m7.convert_image(str(img_path), **settings)
        t1 = time.time()

        # Render back at the same dimensions as the original
        orig_img = Image.open(img_path)
        out_w, out_h = orig_img.size

        rendered_img = td.render_bytes(page, out_w, out_h)

        # Quantise both to palette indices
        cmap_orig, iw, ih = td.load_and_quantise(img_path)
        rendered_arr = np.array(rendered_img, dtype=np.float32)
        flat = rendered_arr.reshape(-1, 3)
        dists = np.sum((flat[:, None, :] - td.PALETTE_RGB[None, :, :]) ** 2, axis=2)
        cmap_rend = np.argmin(dists, axis=1).reshape(ih, iw).astype(np.uint8)

        # Sample sub-pixels
        sp_orig = td.sample_subpixels(cmap_orig, iw, ih)
        sp_rend = td.sample_subpixels(cmap_rend, iw, ih)

        # Compare cell by cell
        matching = 0
        errors = []
        for row in range(td.N_ROWS):
            for col in range(td.N_COLS):
                orig_sp = sp_orig[row * 3:(row + 1) * 3, col * 2:(col + 1) * 2]
                rend_sp = sp_rend[row * 3:(row + 1) * 3, col * 2:(col + 1) * 2]
                if np.array_equal(orig_sp, rend_sp):
                    matching += 1
                else:
                    errors.append((row, col,
                                   orig_sp.flatten().tolist(),
                                   rend_sp.flatten().tolist()))

        match_pct = 100.0 * matching / (td.N_ROWS * td.N_COLS)

        # Categorise errors
        wrong_colour = sum(1 for _, _, o, r in errors
                           if set(o) != {0} and set(r) != {0}
                           and max(set(o), key=o.count) != max(set(r), key=r.count))
        wrong_pattern = len(errors) - wrong_colour

        return {
            'file':          img_path.name,
            'size':          f'{iw}x{ih}',
            'match_pct':     round(match_pct, 2),
            'matching':      matching,
            'error_cells':   len(errors),
            'wrong_colour':  wrong_colour,
            'wrong_pattern': wrong_pattern,
            'convert_s':     round(t1 - t0, 2),
            'first_errors':  errors[:3],
        }

    except Exception as e:
        return {
            'file':      img_path.name,
            'error':     str(e),
            'match_pct': 0.0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0,
                        help='Test only first N images (0 = all)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel worker processes')
    parser.add_argument('--out', default='gallery_results.json',
                        help='Output JSON file')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    all_imgs = sorted(GALLERY_DIR.glob('*.png'))
    imgs = []
    for p in all_imgs:
        try:
            iw, ih = Image.open(p).size
            if 0.7 <= iw / ih <= 1.6:
                imgs.append(p)
        except:
            pass
    if args.limit:
        imgs = imgs[:args.limit]
    print(f'(filtered to {len(imgs)} standard-aspect images from {len(all_imgs)} total)')

    print(f'Testing {len(imgs)} images...', flush=True)

    results = []
    for i, img in enumerate(imgs, 1):
        r = test_one(img)
        results.append(r)
        if args.verbose or i % 100 == 0:
            pct = r.get('match_pct', 0)
            print(f'[{i}/{len(imgs)}] {img.name}: {pct:.1f}%', flush=True)

    # Summary
    valid = [r for r in results if 'error' not in r]
    errored = [r for r in results if 'error' in r]
    pcts = [r['match_pct'] for r in valid]

    print(f'\n=== Summary ===')
    print(f'Tested: {len(valid)}  Errors: {len(errored)}')
    if pcts:
        import statistics
        print(f'Mean match:   {statistics.mean(pcts):.1f}%')
        print(f'Median match: {statistics.median(pcts):.1f}%')
        print(f'Min: {min(pcts):.1f}%  Max: {max(pcts):.1f}%')
        perfect = sum(1 for p in pcts if p >= 99.9)
        good = sum(1 for p in pcts if p >= 80)
        print(f'Perfect (>=99.9%): {perfect}')
        print(f'Good (>=80%):      {good}')
        print(f'Poor (<50%):      {sum(1 for p in pcts if p < 50)}')

    out_path = pathlib.Path(__file__).parent / args.out
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nFull results written to {out_path}')


if __name__ == '__main__':
    main()
