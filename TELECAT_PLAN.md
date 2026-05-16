# telecat — recap and clean-start plan

**telecat** = **Tele**text **C**onditional **A**utoregressive **T**ransformer.
Image-conditional (cross-attention on a ViT encoder), autoregressive (GPT2-style
byte-by-byte decoder), transformer (both halves). "Conditional" is honest about
what we do today; see §8 for "Constrained" as a future direction.

Status: 2026-05-16. Drafted in `image2mode7` while preparing to extract the ML
work into a fresh private repo (`telecat`) that does **not** vendor the DP
solver and uses **only** ethically-sourced data.

---

## 1. What we built

### Architecture (M1)

- `VisionEncoderDecoderModel` = ViT-tiny encoder + GPT2-small decoder + cross-
  attention. ~11.7 M parameters.
- Vocab: 1027 = 1024 byte values + BOS + EOS + PAD.
- Input: 192×256 RGB, normalised to [-1, +1].
- Output: fixed-length 1000-byte sequence (one Mode 7 page).
- Decoder generates greedily, `min_length=max_length=PAGE_LEN+1`, no EOS short-
  circuit (we force a full 1000 bytes).

### Training pipeline

- Modal cloud, single A10G, bf16 autocast, AdamW (0.9, 0.95), weight decay 0.05,
  grad clip 1.0, cosine LR with 500-step warmup, peak 3e-4.
- Webdataset shards: silver = 50 shards × 1000 images × 3 presets = 150 k pairs.
- Detached + spawned for laptop-sleep resilience.
- M1 baseline: 3 epochs, ~50 min, ~$1.50, avg loss 0.699 (final ckpt
  `ckpt-03.pt`). Visually good on horsenburger eval — picks up colours, learnt
  HOLD-graphics behaviour, residual errors on fine detail / control codes.
- M3 (with gold mix): same setup + 5 % gold mix (1909 horsenburger DP-pairs,
  `resampled=True`), avg loss 0.691. Improved noticeably on training-set eval
  (memorisation effect) and modestly on out-of-distribution external set
  (+38.6 byte-match avg over 33 images vs M1).

### Scripts (in `modal_jobs/`)

- `prep_silver.py` — wikiart streaming → DP-encode with 3 presets per image →
  shard.
- `prep_gold.py` — horsenburger PNGs → DP-encode (no preset) → `gold.npz`.
- `prep_gold_shards.py` — `gold.npz` + horsenburger PNGs → webdataset shard.
- `prep_bronze.py` — 112 k real Mode 7 byte streams → render(bytes) → shard.
  **Resulted in failed runs (identity-trap); do not reuse.**
- `train.py` — supports silver-only, silver+bronze mix, silver+gold mix; gold
  uses `resampled=True` to never exhaust mid-epoch.
- `sample.py` — sample on gold horsenburger eval set.
- `sample_external.py` — sample on arbitrary images, DP-encode each on the fly
  as the comparison baseline.
- `model.py` — `build_model()`, IDs, constants.

---

## 2. Results to date

| Run | Mix | Final loss | Byte-match vs gold (in-dist) | Byte-match vs DP (OOD avg, n=33) | Visual verdict |
|-----|-----|------------|------------------------------|----------------------------------|----------------|
| M1  | 100 % silver | 0.699 | 549 (vs proper DP gold) | 660.0 | clear conditioning, residual fine-detail and HOLD-graphics errors |
| M2  | 70/30 silver/bronze | regressed | n/a | n/a | garbled, near-blank — bronze identity trap |
| M2b | 100 % bronze fine-tune | regressed | 77 / 1000 (worst) | n/a | catastrophic forgetting |
| M3  | 95 / 5 silver/gold | 0.691 | 763 (training-set leak) | 698.6 (+5.8 %) | better on teletext-art, slight regression on natural photos |

Other measurement bugs that confounded earlier reads:

- Old `gold.npz` was produced by `teletext_decode.decode_image()` (a
  render-roundtripper), not `image2teletext.convert()` — so all of M1's
  reported byte-match vs gold was vs garbage. Fixed 2026-05-15.
- Byte-match is a brittle signal; use visual comparison + re-rendered pixel
  similarity instead. See `memory/project_eval_metrics.md`.
- The DP "gold" comparison in `sample_external.py` uses **no preset**, so it
  bypasses gamma / contrast / saturation / sharpening. Photographic images
  (e.g. Mona Lisa) get a desaturated low-detail "gold" that doesn't match what
  the CLI / UI would produce. To be addressed in clean run.

---

## 3. Datasets — what we have and what's usable

### Eval-only (never in training)

| Dataset | Source | Reason |
|---------|--------|--------|
| **horsenburger gallery** (1909 PNGs) | `gallery/gallery_urls.txt` → wixstatic.com | Bulk-scraped from the artist's own site without rate-limiting or a User-Agent identifier. `horsenburger.com/robots.txt` is actually permissive (`User-agent: * / Allow: /`, no general crawl-delay), so this didn't strictly violate robots — the issue is the *manner* of the scrape. Acceptable as an **eval / sample-comparison set** (we're just running inference on them locally, not redistributing), but **not** as training data. |
| **`gold.npz` / `gold-0000.tar`** on Modal volume | Derived from horsenburger | Eval-only material. Can stay on the volume for sample-comparison runs; not eligible to be mixed into any future training shard. |

### Discard

| Dataset | Source | Reason |
|---------|--------|--------|
| **`ckpt-m3-*.pt`** | Trained with horsenburger gold mixed into the training stream | Tainted as a *trained* artifact. Don't publish or build on. (M1 ckpt-03 is unaffected — pure silver, never saw horsenburger.) |
| **`samples_m3` / `samples_external_m3`** | Outputs of the tainted M3 ckpt | Useful for retrospective debugging only. |

### Keep, with caveats

| Dataset | Source | Status | Caveat |
|---------|--------|--------|--------|
| **16colors images** (~3000) | `scrape_16colors_teletext.py` → 16colo.rs | OK — respects robots.txt, 1 s+ delay, per-pack zips | Includes ~30 `*HORSENBURGER*`-attributed entries; these are kept because the artist published them on 16colors themselves (artist-sanctioned distribution). |
| **16colors `pages/`** (~3000 bins) | Produced via `decode_16colors_to_bytes.py` (the buggy round-tripper) | **Not valid data** — discard | Must be regenerated via `image2teletext.convert()`. |
| **wikiart silver source** | huggan/wikiart on HF | OK — public ML dataset | None. |
| **`silver-*.tar` shards** on Modal volume | DP-encoded wikiart with 3 presets | Usable as-is for re-training, OR regenerate with single preset (see lesson 5). | Targets are inconsistent (3 presets per image). |
| **`computer_legacy/pages/`** | `fetch_computer_legacy.py` → computer-legacy.com | OK — User-Agent, 1 s rate-limit, public catalogue. T42 captures of broadcast Mode 7. | **Bytes-only** — no source image. Only useful as decoder-LM pretraining or grammar prior. |
| **`teletextart/pages/`** (dan archive) | `dan_to_pages.py` from teletextart.co.uk dan.zip | OK — single zip download. | **Bytes-only** — same constraint as above. |
| **`test/pngs/`** (21 traditional test images) | Local, classic CV references | OK for OOD eval | No source for some, treat as eval-only. |

### Net assets going into telecat

- ~2980 real teletext-art image+bytes pairs (16colors as-is, freshly DP-encoded).
- ~150k wikiart photo+bytes pairs (or rebuilt as ~50k single-preset pairs).
- ~110k+ real Mode 7 byte streams (computer-legacy + teletextart) for optional
  decoder-LM pretraining.
- 21 fixed external test images for OOD eval.

---

## 4. Lessons learned (constraints on the clean run)

1. **byte-match against a single reference is brittle.** Many valid encodings
   render to similar pages. Primary signal must be visual; byte-match is
   tertiary diagnostic. Re-rendered pixel similarity (`render_bytes(pred)` vs
   input) is a better automated proxy.
2. **`render(bytes) → bytes` is an identity-trap.** Never use as supervised
   training pairs. The model abandons the hard task in favour of the easy one
   and collapses on real photos.
3. **`wds.RandomMix` exhausts on the shortest stream**, killing the entire
   loader mid-epoch and silently undertraining. Small auxiliary streams must
   use `resampled=True` so they cycle forever.
4. **`wds.WebDataset` with `nodesplitter=split_by_node` + multi-worker
   DataLoader will give some workers zero shards** if shard count < worker
   count. Use `empty_check=False`.
5. **Silver targets with 3 presets per image are inconsistent** — same input,
   three different "correct" outputs. The deterministic model can only produce
   one, so it ends up at something like a learned average. For the clean run,
   prefer **one preset per image** (probably `photo`) for consistent targets.
6. **`image2teletext.convert(img)` with no preset bypasses all preprocessing**
   (gamma, contrast, saturation, sharpening). For photo-style sources the
   resulting gold is desaturated and low-detail. Either pick the preset
   per-image-class or use a sensible default like `photo`.
7. **Gold-mix shifts the model toward the gold distribution.** Mixing in 1909
   teletext-art images improved teletext-art predictions and slightly hurt
   natural photos. The mix sets the model's aesthetic bias, so curate it
   accordingly.
8. **bf16 + A10G + batch 32 + lr 3e-4 + 3 epochs is a stable recipe** for 150 k
   samples at this model size. ~50 min wall, ~$1.50.
9. **Detached + spawn keeps Modal alive across laptop sleep**, but `--spawn`
   alone already detaches; combining with `--detach` is fine but redundant.

---

## 5. Proposed telecat repo

### Principles

- **No DP solver source vendored.** `image2teletext` and `teletext_decode`
  become external pinned dependencies (either pip-installable, or a git
  submodule pinned to a SHA from `image2mode7`). Telecat only knows the
  model + training + eval.
- **Datasets are listed, licensed, and provenance-documented** in
  `DATASETS.md`. No image is included whose source isn't recorded.
- **Private from day one.** Public history of image2mode7 is unaffected.
- **All preprocessing recipes are recorded** in the dataset prep scripts and
  pinned in version control. No "build silver with 3 presets" lost-knowledge
  surprises.

### Suggested layout

```
telecat/
  README.md              brief — what this is, hardware, how to train
  DATASETS.md            full provenance: source, license, count, prep recipe
  ML_NOTES.md            this document, trimmed of the cleanup parts
  pyproject.toml
  src/telecat/
    model.py             build_model, constants
    train.py             pure-PyTorch training loop (no Modal)
    sample.py            local sampling utility
    eval.py              metrics: byte-match, re-rendered pixel sim, validity
  modal/
    image.py             shared Modal image / Volume / App definitions
    prep_silver.py       wikiart → DP-encode (single preset) → shards
    prep_gold.py         16colors images (Horsenburger filtered) → DP-encode → shard
    prep_lm.py           OPTIONAL: computer-legacy + dan bytes → LM shards
    train.py             Modal entrypoint wrapping src/telecat/train.py
    sample.py            Modal entrypoint
  data/
    README.md            tells reader where data lives (Modal volume) — not in repo
    test/pngs/           the 21 OOD evals (small, fine to commit)
```

### What does NOT come over

- The DP solver source.
- `gallery/` directory and `gallery_urls.txt`.
- `prep_bronze.py`, anything bronze-related.
- `gold.npz`, current `gold-*.tar`, `ckpt-m3-*.pt` derived from horsenburger.
- The ML_TRAINING_NOTES.md (its M1-M3 narrative depends on horsenburger).

---

## 6. Proposed clean training run (M4 — first real telecat baseline)

### Data prep (Modal CPU)

1. **Re-scrape 16colors if needed** (existing local copy may suffice). Confirm
   `scrape_16colors_teletext.py` ran cleanly and recently.
2. **Keep all 16colors entries**, including Horsenburger-attributed ones — the
   artist published them on the 16colors platform themselves. No filename
   filter needed. (Only the wixstatic-scraped 1909-image gallery is excluded,
   and those don't appear in the 16colors corpus.)
3. **DP-encode 16colors** with `image2teletext.convert(img, preset='art')` →
   target ~2980 pairs. Write to `gold-0000.tar` etc.
   - `preset='art'` is correct for existing teletext-art sources: dithering
     **off** (Floyd-Steinberg would speckle flat colour regions and destroy
     the hand-crafted aesthetic), flat-colour quantisation, snap-to-palette.
4. **Re-build silver from wikiart** with a single preset (`photo`) →
   `silver-*.tar`. Same shard size (1000/shard). Same 50 k target.
   - `preset='photo'` is correct for photographic sources: Floyd-Steinberg
     dithering **on** to render smooth tone gradients, modest sharpening,
     saturation/contrast boost to exploit the Teletext palette.
   - Single target per input (vs M1's 3 presets per image) gives the model a
     consistent objective and removes the averaging-across-styles problem.
5. **Hold out 200 images from 16colors gold** as `gold_eval-*.tar`, never seen
   in training. Use for in-distribution byte-match.
6. **Keep `test/pngs/` (21 images)** as the OOD eval set, never seen.
7. **Horsenburger gallery (1909) is available as a third eval set** —
   sample-only, comparison only, never mixed into training shards. Useful
   because it's a coherent single-artist style we can read trends on.

### Training

- Same architecture as M3 (ViT-tiny + GPT2-small, 11.7 M).
- Same recipe: bf16, AdamW (0.9, 0.95), wd 0.05, grad clip 1.0, cosine LR
  warmup 500, peak 3e-4, 3 epochs, batch 32, 4 workers.
- Silver + gold mix at 5 %, gold resampled, `empty_check=False`.
- Detached + spawn.

### Evaluation (run after each ckpt)

- **In-distribution byte-match** vs held-out 16colors gold (n=200).
- **OOD byte-match** vs DP-encoded test/pngs (n=21, both `art` and `photo`
  preset variants for fairness).
- **Visual HTML grid** for in-dist + OOD.
- **Re-rendered pixel MSE**: render(pred) vs input, averaged.
- **Stream validity**: does pred parse as well-formed Mode 7? (boolean count.)

### Expected first comparison

Once M4 has finished, compare against M1 only (M3 is tainted). If M4 byte-match
vs DP on OOD beats M1 (660), the clean pipeline is a real upgrade. If it's
similar, we know the horsenburger mix in M3 was doing most of the work and we
need a different aesthetic anchor (e.g. larger 16colors set, or larger silver).

---

## 7. Future data sources (not for M4, parked for later)

| Source | Status | Notes |
|--------|--------|-------|
| **demozoo.org** teletext-tagged graphics | Blocked by robots | `/search/` is `Disallow: /` for all UAs, and `/search/?q=teletext&category=graphics` is the only discovery path. `/tags/teletext/` and a "Teletext" platform don't exist; individual `/productions/<id>/` pages are fetchable (10 s crawl delay) but we have no compliant way to enumerate IDs. **Admins are known to the user but not yet sympathetic** — revisit later, either via manual ID curation, or by re-asking admins for a dump / whitelist once the project has more to show for itself. |
| **teletextart.co.uk** artist tag list and individual artist pages | Robots-permissive but consent-sensitive | `robots.txt` only blocks `/wp-admin/`, no crawl-delay. Technically scrapable. The site is a *curated showcase* of individual artists' work — copyright stays with the artist, and inclusion in a group feature isn't implicit consent for ML training (unlike 16colors, where each pack is an explicit artist-driven distribution). Right path: per-artist permission, not bulk scrape. We already have one author's archive (`datasets/teletextart/dan.zip`, made publicly downloadable by Dan Farrimond) — that one is fine; widening would mean reaching out to other featured artists individually. |
| (add more here as they come up) | | |

## 8. Future capability directions (not for M4)

| Direction | What it is | Why it's interesting |
|-----------|------------|----------------------|
| **Constrained decoding** (earn the "C" of CAT) | Mask the logits at generation time so the model can only emit byte sequences that form valid Mode 7 streams (e.g. graphics-mode bytes only after the right control codes, foreground-colour bytes only in legal positions, no impossible HOLD-GFX transitions). Requires encoding the Mode 7 grammar as a transition table over decoder state. | Directly attacks the failure modes seen in M3 (block repetition, hallucinated colours, broken HOLD-graphics) by making them generation-time impossible rather than something the model has to learn to avoid. Stream-validity becomes a hard guarantee instead of a metric. Telecat's "C" becomes literally accurate (Constrained Autoregressive Transformer) rather than just Conditional. Non-trivial: the grammar definition has to be precise and the masking has to be fast enough to not bottleneck inference. |
| **Decoder-LM pretraining on real teletext bytes** | Pre-train just the GPT2 decoder on the ~110k computer-legacy + dan byte streams as a pure language-model task (no image input), then freeze/warm-start it for full image-conditioned training. | Teaches Mode 7 grammar from genuinely-authored streams without the bronze identity-trap. Could be combined with constrained decoding above. Likely worth trying after M4 establishes a clean baseline. |
| **Bigger model — ViT-small + GPT2-medium (~50M)** | Same recipe, 4× the parameters. | The architectural-looking failures (block repetition, foreground-black struggle) may be capacity-bound. Worth testing once data pipeline is settled. |
| **Re-rendered pixel similarity as a real metric** | LPIPS or pixel MSE between `render_bytes(pred)` and the original input image, averaged across an eval set. Currently informal / by-eye. | Gives a single scalar that correlates with visual quality, unlike byte-match. Needed to make hyperparameter sweeps tractable without manual review every time. |

## 9. Open questions before starting

1. ~~**Silver preset choice.**~~ **Decided 2026-05-16:** single `photo`
   preset per wikiart image. Dithering on (right for photos), single target
   per input avoids the multi-preset inconsistency.
2. ~~**Gold preset for 16colors.**~~ **Decided 2026-05-16:** `art` preset.
   Dithering off (right for hand-crafted teletext art), flat-colour snap.
3. ~~**Eval preset for `test/pngs/`.**~~ **Decided 2026-05-16:** render
   both `art` and `photo` golds per image (small set, cheap), and show both
   columns in the HTML grid. Lets us see whether the model is closer to one
   or the other per-image and gives a fairer ceiling than picking one.
4. ~~**Decoder-LM pretraining on computer-legacy + dan bytes.**~~
   **Decided 2026-05-16:** defer. These datasets are bytes-only (no images),
   and the previous "discount" specifically rejected the *bronze* use
   (`(render(bytes), bytes)` supervised pairs — identity trap). Pure LM
   pretraining of the decoder on the same bytes is a structurally different
   technique (no fake pairs, just grammar prior), so it's not ruled out by
   the bronze finding. But it adds a pretrain phase + warm-start complexity
   for uncertain benefit, and constrained decoding (§8) targets the same
   failure modes more directly. Revisit only if M4 shows persistent
   block-repetition / invalid-control-sequence symptoms.
5. ~~**Modal volume hygiene.**~~ **Decided 2026-05-16:** delete tainted
   ckpts (`ckpt-m2-*.pt`, `ckpt-m2b-*.pt`, `ckpt-m3-*.pt`). Keep downloaded
   / converted datasets that are still relevant so we don't redo expensive
   prep work: horsenburger PNGs + gold.npz (eval-only use is still allowed),
   16colors source images, computer-legacy + dan byte archives, the
   `external_test/` upload. Silver shards (3-preset) and the gold-0000.tar
   webdataset are now superseded by the M4 single-preset rebuild plan —
   decide whether to delete or just let them be overwritten when M4 prep
   runs. Keep the same `teletext-m1` volume for now; clean enough after the
   deletes that a new `telecat` volume isn't worth the migration friction.
6. ~~**DP solver dependency mechanism.**~~ **Decided 2026-05-16:** pip
   install via git URL. Work breakdown:
   - Add `pyproject.toml` to `image2mode7` that exposes `image2teletext` and
     `teletext_decode` as importable top-level modules (they currently are,
     they just lack packaging metadata).
   - Tag a release (`v0.1.0`) on the public repo so telecat can pin to it.
   - In telecat's `pyproject.toml`:
     `image2teletext @ git+https://github.com/kieranhj/image2mode7@v0.1.0`.
   - Modal images: install via the git URL in the `image.pip_install(...)`
     chain — same one-liner cache behaviour as any pip dep.
   - Local dev: `pip install -e ../image2mode7` for instant feedback when
     iterating on solver + ML together.

---

**All open questions resolved as of 2026-05-16.** Ready to start the
extraction work whenever you give the go-ahead. Recommended first concrete
actions, in order:
  1. Run the volume cleanup deletes (§10 — needs your final confirm on the
     optional items first).
  2. Add `pyproject.toml` + tag `v0.1.0` on this (image2mode7) repo.
  3. Create empty `telecat` private repo; first commit = this plan trimmed
     + `DATASETS.md` + skeleton.
  4. Port `prep_silver` / `prep_gold` / `train` / `sample` into telecat
     using the new conventions (single preset, art-preset gold, resampled
     mixer, both eval previews).
  5. Launch M4.

---

## 10. Cleanup checklist before extraction

- [ ] Delete `ckpt-m3-*.pt`, `ckpt-m2*.pt` from `teletext-m1` Modal volume
  (trained-with-horsenburger artifacts; tainted as published models).
- [ ] Keep `gold.npz`, `gold-*.tar`, `horsenburger/` PNGs on the volume —
  eval-only use is still allowed.
- [ ] In telecat training scripts, hard-code the rule that horsenburger
  shards must never appear in the training mix (allow them only in eval
  scripts).
- [ ] Make final note that `gallery_urls.txt` and `horsenburger/` were
  excluded.
- [ ] On the day of extraction: take a snapshot of the wikiart silver shards
  (these are reproducible from the scraper but the shards have been validated
  to work).
- [ ] Decide on dependency mechanism for `image2teletext` (Q6).
- [ ] Decide repo visibility and host (likely github private).
- [ ] First telecat commit: this document (trimmed), `DATASETS.md`,
  `pyproject.toml`, empty module skeleton.
