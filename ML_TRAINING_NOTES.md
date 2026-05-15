# Training a model to emit Mode 7 / Teletext byte streams

A technical sketch for replacing (or complementing) the hand-written DP solver
in `image2teletext.py` with a learned model that takes an image in and emits
the 1000-byte Mode 7 page directly.

---

## 1. Framing the problem

Mode 7 output is a **fixed-length discrete sequence**: 25 rows × 40 bytes =
1000 tokens, vocabulary ≤ 256. That makes it look superficially like an image,
but it is not — it is a *program* in a tiny stateful language (the SAA5050
control codes), so:

- A pixel-space diffusion model that paints a 40×25 "image of bytes" will
  produce visually plausible pages that **do not decode**, because byte 0x91
  in column 5 changes the meaning of every cell to its right on that row.
- The natural framing is **conditional sequence generation**: image → 1000
  tokens, autoregressive or non-autoregressive, with the loss computed in
  *byte space* and (optionally) a perceptual loss computed by running the
  rendered output through `teletext_decode.py`'s renderer back to RGB.

There are three credible families of model:

| Family | Description | Fit |
|---|---|---|
| **Image-conditioned autoregressive transformer** | ViT/CNN encoder → causal decoder over 1000 byte tokens. Equivalent to image captioning with a 256-token vocab and fixed length. | **Best fit.** Naturally enforces stream semantics: each byte is generated knowing every byte to its left, including control codes. |
| **Discrete diffusion (D3PM / MaskGIT)** | Iteratively denoise a 25×40 grid of byte tokens. | Plausible. Faster sampling than AR. Risk: parallel unmasking does not respect left-to-right state, so control-code semantics must be learned the hard way. |
| **Continuous latent diffusion + decoder head** | Diffuse in a low-dim latent, decode to bytes with a small AR head. | Overkill at this output size. |

The rest of this doc focuses on the **AR transformer** path with notes on
where discrete diffusion would diverge.

---

## 2. Training data

This is the hard part. You need (image, 1000-byte page) pairs where the page
is *good* teletext art for the image. There are four sources, in order of
quality:

### 2a. Human-authored pages (gold)

The `gallery/` directory already lists ~1900 URLs from horsenburger.com /
teletextart.co.uk. After running `teletext_decode.py` over the rendered PNGs
you get ~1900 (rendered_PNG, byte_stream) pairs. The "image" here is the
rendered teletext itself, not a natural photo — useful for an
**autoencoder/identity** pretraining task ("given a teletext-looking image,
emit bytes that reproduce it") but not directly for "natural photo →
teletext".

To bridge that gap, generate a **photo-like conditioning image** from each
gold page by upscaling, blurring, slight colour jitter, and resaturation —
approximating "what a photograph of this scene might look like before
quantisation". Crude, but it teaches the model that many photo-space inputs
collapse to one canonical teletext page.

### 2b. Solver-generated pairs (silver)

Run `image2teletext.py` over a large image corpus (LAION subset, COCO,
Unsplash-lite, Open Images) with a variety of preset/flag combinations:

```
for img in corpus:
    for preset in [photo, art, vivid, graphic, retro, flat, smooth]:
        emit (img, image2teletext(img, preset).bytes)
```

This is cheap — a few seconds per image on CPU — and gives you essentially
unlimited supervised data, but the ceiling is the solver's own quality. Treat
this as **pretraining**, then fine-tune on the gold set.

Target volumes:
- Pretraining (silver): 200k–1M pairs. ~1M is easy at 7 presets × 150k images.
- Fine-tune (gold): 1.9k pages × ~5 photo-augmentation variants ≈ 10k pairs.

Storage: 1M × (256×192 RGB jpeg ≈ 15 KB + 1000 bytes) ≈ **16 GB**. Fits
locally easily.

### 2c. Decoded-from-render pairs (bronze)

For any teletext page (BBC archive `.ssd` files, edit.tf shared pages, MAME
captures), render to PNG and decode. Adds maybe a few thousand more pages.

### 2d. Synthetic curriculum

Programmatically generate pages with known structure — solid colour
rectangles, gradients, simple silhouettes, text — to teach the model the
control-code grammar before exposing it to messy natural images. ~50k of these
costs nothing.

### 2e. Current on-disk inventory

Datasets actually fetched, under `datasets/`. Strategic tier (gold / silver /
bronze) is per §2a-d; "format" is what the file actually is, not what the
training pipeline ultimately consumes.

| Path | Source | Format | Count | Tier | Purpose |
|---|---|---|---|---|---|
| `datasets/16colors_teletext/images/` | [16colo.rs](https://16colo.rs/tags/content/teletext) `tags/content/teletext` | PNG/GIF renders | 2,983 | gold (authored, source) | Hand-authored Mode 7 art, pixel-exact renders of artists' teletext intent. The "what good teletext looks like" corpus. |
| `datasets/16colors_teletext/pages/` | decoded from `images/` via `decode_16colors_to_bytes.py` | 1000-byte BBC Mode 7 | 2,980 | gold (authored) | Decoder is graphics-only, so any text portions are reconstructed as approximate graphics. JPEG sources add quantisation noise. Use these for the "graphics aesthetic" half of the corpus, not text-heavy authored pages. |
| `datasets/teletextart/pages/` | teletextart.co.uk `dan.zip` (Jason Robertson archive) | 1000-byte BBC Mode 7 | 4,301 | gold (broadcast) | Real UK broadcast teletext (BBC1, GMTV, Sky One, TCC, TVAM, 1991-93), already in our pipeline's target byte format (control codes 0x80–0x9F). Bytes are byte-exact ground truth; rendered images are noisy (VHS recoveries). Use as direct supervision for byte-level decoder. |
| `datasets/teletextart/dan/**/*.t42` | dan.zip | T42 packet stream | 18 streams | source | Raw broadcast captures backing the pages above. Reprocess with `teletext` CLI (`pip install git+https://github.com/ali1234/vhs-teletext`) for alternative recoveries — sometimes finds pages a single squash misses. |
| `datasets/computer_legacy/zips/` | [computer-legacy.com](https://computer-legacy.com/teletext.html) (Steve Horsley archive) | T42 streams (zipped) | 104 zips, 38 MB | source | High-quality VHS recoveries with **5-star quality ratings** in the catalogue. Filtered to 4-5★ + all hand-finalised. Mostly UK broadcast 1990s-2000s (BBC1/2, C4, ITV regions, Sky). Each zip = one capture (15 min - 3 h). |
| `datasets/computer_legacy/pages/` | extracted from `zips/` via `extract_t42_pages.py` | 1000-byte BBC Mode 7 | 108,108 pages | gold (broadcast) | Largest byte-stream corpus we have. Extracted using the `teletext` library's proper Hamming-decoded pagination. Includes lots of subpage variants and intra-corpus duplicates (page 100 appears in many recoveries) — dedup by sha256 before training. |
| `datasets/teletext_assets/` | [al.zerostem.io](https://al.zerostem.io/~al/teletext/) | CSS + TTF | 13 files | tooling | Canonical teletext rendering assets (Alistair Buxton's set). Drop alongside `teletext html` output for visual verification of byte streams. Note: `.row{display:block}` patch needed in `teletext.css` because the `teletext html` CLI emits `<span class="row">` without newline separators. |

Companion to the strategic tiers in §2a–d:

- **Authored gold (16colo.rs)** is the cleanest signal for "Mode 7 aesthetics"
  but only ~3k pages, biased toward graphics-heavy art (portraits, scenes)
  rather than typical broadcast layouts.
- **Broadcast gold (dan.zip pages)** is 4× larger and gives us text-heavy
  pages with menus, navigation, and the canonical magazine structure — but
  has VHS-recovery corruption baked into the bytes.
- **Together** they cover the two ends of the teletext distribution:
  artistic-graphical and informational-textual.

Planned additions (not yet fetched):

- al.zerostem.io: `0047.0.d.t42.xz` (~9.6 MB compressed), `0047.0.s.t42.xz`
  (squashed), `bbc1-1.s.t42` (1.2 MB), `some-random-pages.zip` (12 MB).
  Likely yields several thousand more broadcast pages once decoded.
- The ~73 dated edition folders on archive.teletextart.co.uk (HTML format,
  parseable via the `f0–f7 b0–b7 dh nx` CSS-class scheme). Lower priority —
  more effort to parse and likely overlaps with dan.zip content.

### 2f. Pickup notes (next session)

Current state: **~115,389 byte-stream pages** on disk across three sources
(authored 16colo.rs, broadcast dan.zip, broadcast computer-legacy). The
renderer in `teletext_decode.py` was extended to handle alpha characters
via `teletext2.ttf` glyphs, so byte streams now render legibly for both
graphics-heavy authored art and text-heavy broadcast pages.

**Resume here, in order:**

1. **Fetch al.zerostem.io raw files** (~25 minutes, ~23 MB).
   - `curl` the four files: `0047.0.d.t42.xz`, `0047.0.s.t42.xz`,
     `bbc1-1.s.t42`, `some-random-pages.zip` from `https://al.zerostem.io/~al/teletext/`
   - Decompress the `.xz` files (`xz -d`).
   - Run `extract_t42_pages.py --in-dir datasets/al_zerostem/zips --out
     datasets/al_zerostem/pages` after wrapping the loose `.t42` files in
     zips (or extend `extract_t42_pages.py` to accept loose `.t42` too).
   - `some-random-pages.zip` may have a different internal layout; probe
     before processing.

2. **Phase 3: Consolidate the byte corpus.** The four sources currently live
   in their own directories with inconsistent filename conventions and
   provenance schemas. Build a single unified corpus:
   - One directory `datasets/corpus/pages/` containing every 1000-byte page,
     filename `<sha256[:16]>.bin`.
   - Master manifest `datasets/corpus/manifest.json` keyed by sha256, recording
     every source filename, provenance (source dataset, channel, date, rating,
     etc.), and quality tier.
   - Sha256 dedup will collapse the 108k computer-legacy pages dramatically
     (page 100 from many BBC1 captures will repeat). Expect 30-50k unique
     pages after dedup.

3. **Phase 3b: Quality split.** Tag each page in the manifest with a
   coarse quality tier:
   - `gold-authored`: 16colo.rs decoded pages (artist intent, but lossy
     decoder output)
   - `gold-finalised`: computer-legacy `kind=finalised` pages (hand-edited)
   - `gold-broadcast`: computer-legacy 4-5★ + dan.zip pages
   - `silver-broadcast`: computer-legacy 3★ pages
   - Drop anything that fails sanity checks (e.g. all-zeros, all-spaces,
     no displayable rows).

4. **Phase 3c: Visual sanity-check at scale.** Render N=100 random pages
   from each tier into a contact-sheet HTML page using the new
   alpha-aware `render_bytes`. Helps catch any source-specific decoding
   bugs (e.g. EBU vs BBC control code conversion errors) before training.

5. **Phase 4: Renderer polish (optional).** Two small extensions when
   needed:
   - **Double-height (DH, 0x8D)**: spec says affected cell and the cell
     directly below render at 2x vertical font scale, with the row below
     blank. Currently single-height — affects header/title rows. ~30 lines.
   - **Flash (0x88) / conceal (0x98)**: visual indicators in static renders.
     Low priority unless used for training.

6. **Phase 5 (the actual project): connect to training story.**
   - Decide on framing: pure byte-stream language model first (sequence-only
     pretraining on the corpus, no image conditioning), then add image
     conditioning. OR jump straight to image-conditioned per §3-§9.
   - Generate input photos: render+augment for the autoencoder pretrain task,
     plus a real-photo corpus for the photo→teletext task (silver via solver).
   - Modal pipeline (§9): `prep_corpus` (new), `prep_silver`, `train`,
     `sample`, `eval`. The `prep_gold` from §9 step 2 is replaced by reading
     from `datasets/corpus/`.

**Tools/scripts already in place:**

| Script | What it does |
|---|---|
| `scrape_16colors_teletext.py` | Per-pack zip scrape from 16colo.rs (already run; cached zips in `datasets/16colors_teletext/zips/`) |
| `decode_16colors_to_bytes.py` | PNG/GIF → 1000-byte page via existing decoder |
| `dan_to_pages.py` | Per-page `.bin` extraction from dan.zip |
| `fetch_computer_legacy.py` | Filtered downloader for computer-legacy.com (4-5★ + finalised) |
| `extract_t42_pages.py` | T42 stream → 1000-byte pages via the `teletext` library (works on any zip-of-t42 directory) |
| `teletext_decode.render_bytes` | Now handles alpha glyphs + graphics; canonical 480×500 output |

### Tokenisation

Vocabulary is just the 256 byte values + a few specials (`<bos>`, `<eos>`,
`<pad>` — though length is fixed so `<eos>` is optional). No BPE needed. Each
byte gets its own embedding. Sequence length is exactly **1000** (or 1025
with row-separator tokens, which empirically helps the model learn the
"control codes do not cross row boundaries" rule).

---

## 3. Model architecture

A reasonable starting point — and one that is small enough to actually train
on the hardware in §5:

```
Image encoder:    ViT-Tiny or ConvNeXt-Atto, ~5–10 M params
                  Input: 256×192 RGB (matches Mode 7 pixel aspect ratio 1.2)
                  Output: 96 patch tokens × 256 dim

Cross-attn bridge: 2 layers, 256 dim

Decoder:          12-layer causal transformer, 256 dim, 4 heads
                  Vocab: 256 + specials
                  Position: learned, max len 1025
                  Total: ~15–20 M params

Total:            ~25–30 M params, ~120 MB fp32 / 60 MB fp16
```

This is roughly the size of a small image-captioning model (think OFA-tiny or
TrOCR-small). It is small because the task is small: the output space has
about 8 bits of entropy per token and 1000 tokens, so the upper bound on
information content per page is ~1 KB. A 30M-param model is wildly
over-parameterised for that, which is fine — it leaves capacity for learning
the conditioning.

### Loss

Two terms, weighted:

1. **Token cross-entropy** against the target byte stream (teacher forcing).
2. **Rendered-image perceptual loss**: differentiably-render the predicted
   bytes (argmax / Gumbel-softmax over the byte distribution → palette indices
   → 80×75 sub-pixel grid) and take MSE or LPIPS against the input image. This
   is the same error metric the DP solver minimises, just applied as a
   training signal. Implementing the renderer in PyTorch from
   `teletext_decode.py` is ~200 lines and makes a large quality difference.

---

## 4. Frameworks and libraries

Pick the boring stack — this is a small model, you don't need anything exotic:

| Layer | Recommendation |
|---|---|
| Core | **PyTorch 2.5+** with `torch.compile` |
| Model code | **HuggingFace `transformers`** (use `VisionEncoderDecoderModel` as scaffolding — already does ViT-encoder + causal-decoder + cross-attn) or write 300 lines yourself |
| Training loop | **`accelerate`** for mixed precision + device placement; **`lightning`** if you want callbacks/logging out of the box |
| Data | **`webdataset`** for streaming the silver pairs from disk as tar shards |
| Logging | **Weights & Biases** or `tensorboard` |
| Inference | Same model exported to **ONNX** or **OpenVINO IR** for fast CPU/iGPU inference |

For discrete-diffusion alternatives: `lucidrains/d3pm-pytorch` or write it
from the MaskGIT paper in ~500 lines.

### The Intel-specific bits (relevant for §5)

You do **not** have an NVIDIA GPU on this laptop. Your options for training
acceleration are:

- **Intel Extension for PyTorch (`intel_extension_for_pytorch`)** — adds an
  `xpu` device that targets the Arc 140T iGPU via SYCL/Level Zero. Works for
  fp16 and bf16. Maturity is improving but not on par with CUDA — expect to
  hit the occasional unsupported op and fall back to CPU.
- **OpenVINO training extensions** — primarily inference-focused, but the
  NNCF compression toolkit supports QAT.
- **Pure CPU with `torch.compile` + bf16** — the Ultra 7 255H has AVX2 and
  AMX-like extensions; for a 30M model this is slow but workable for
  fine-tuning, not for from-scratch pretraining on 1M pairs.

---

## 5. Hardware assessment for *this* laptop

Your machine, as detected:

- **CPU**: Intel Core Ultra 7 255H, 16 cores, 2.0 GHz base
- **GPU**: Intel Arc 140T iGPU, ~2 GB dedicated VRAM, ~16 GB shared
- **RAM**: 32 GB
- **OS**: Windows 11

### What is feasible locally

| Task | Verdict | Notes |
|---|---|---|
| **Inference** of the trained 30M model | ✅ Easy | <100 ms/page on iGPU via OpenVINO; ~300 ms on CPU. Real-time UI feasible. |
| **Fine-tuning** on the 10k gold set, 5–10 epochs | ✅ Plausible | Estimate 4–12 hours on the Arc iGPU with bf16, batch 8–16. CPU-only: 1–3 days. Memory headroom is fine (model + optimizer states ≈ 1 GB at fp16/AdamW). |
| **Pretraining** on 200k silver pairs, 5 epochs | ⚠️ Marginal | ~3–7 days on the iGPU if IPEX cooperates, ~2 weeks on CPU. Doable but unpleasant — you'll want the laptop on mains and you won't be using it for much else. |
| **Pretraining** on 1M silver pairs | ❌ Don't | Would take 2–4 weeks and you'd be fighting thermals. Rent a GPU. |
| **Discrete diffusion** training | ❌ Harder | More forward passes per step, larger memory, less mature on Intel XPU. CUDA strongly preferred. |

### Bottlenecks specific to this laptop

1. **Thermal throttling**. The Zenbook 14 is a thin-and-light; sustained
   100% GPU+CPU load drops clocks within minutes. Expect 30–50% of peak
   throughput on long runs. An external cooling pad helps surprisingly much.
2. **iGPU memory is shared**. You will see "16 GB" reported but in practice
   you have ~10–12 GB of usable headroom before Windows starts swapping
   compositor textures. Keep batch size modest.
3. **Driver maturity**. IPEX + Arc on Windows is functional in 2026 but
   crash-prone for long jobs. Checkpoint every epoch. Consider WSL2 + Linux
   IPEX builds — generally more stable.
4. **No bf16 acceleration on the iGPU for all ops** — some ops fall back to
   fp32. Watch the IPEX op-coverage table for the release you're using.

### Recommended split

- **Develop, debug, and overfit on 100 samples**: locally. CPU is fine for
  this; you want fast iteration.
- **Fine-tune on the gold set**: locally on the iGPU, overnight.
- **Pretrain on silver**: rent. ~$20–40 of A100 or L40S time on RunPod /
  Lambda / Modal will do 1M pairs × 5 epochs in well under a day. Save the
  checkpoint, fine-tune locally.

This split is the actual sweet spot for a project this size: cloud for the
embarrassingly-parallel pretraining job that runs once, laptop for everything
iterative.

---

## 6. Evaluation

You already have most of the eval harness — `gallery/test_gallery.py` and
the rendered-PNG-vs-source comparison in the existing test suite. For a
trained model add:

- **Byte accuracy** (token match) on a held-out gold subset.
- **Rendered MSE / LPIPS** of model output vs. ground-truth rendered page.
- **Stream validity rate**: fraction of generated pages that decode without
  producing garbage (control codes used in implausible positions, holds
  without prior graphics, etc.). A 1.0 here is non-trivial and is the main
  argument for AR over diffusion.
- **A/B vs. DP solver**: human eval on N=50 pairs. The bar to beat is
  `--preset art` plus `--edge-weight 3 --sep` — that is already very good,
  and a small model fine-tuned on the gold set should mainly differ in
  having a more *artistic* (less faithful) output, which is the whole point.

---

## 7. Suggested first milestone (1 weekend of work)

1. Decoder-render 1900 gallery PNGs into `(rendered_png, byte_stream)` pairs
   using existing `teletext_decode.py`. Already most of the way there.
2. Generate 50k silver pairs from a small image corpus with 3 presets.
3. Train a 10M-param VisionEncoderDecoder for 3 epochs on the 50k silver
   set, locally on iGPU. Budget: one overnight run.
4. Evaluate byte accuracy and stream validity. If the latter is >95% and
   rendered output is recognisable, the architecture works and it's worth
   investing in the gold-set fine-tune and the cloud pretraining run.

If milestone 1 produces garbage, the most likely culprits are (a) you fed it
silver-only and overfit to solver artefacts, or (b) the cross-attention
bridge is too small — bump it to 4 layers before giving up.

---

## 8. Doing the whole thing on a rented machine (recommended)

If you'd rather not nurse the laptop through multi-day jobs, the simplest
end-to-end remote workflow is:

### 8a. Pick a provider

For a one-person, one-model project the right tier is **on-demand single-GPU
notebooks/instances**, not Kubernetes-flavoured MLOps platforms. Ranked by
how little setup they need:

| Provider | What you get | Cost (Apr 2026, approx) | Friction |
|---|---|---|---|
| **Lightning AI Studios** | Persistent VS Code in the browser, mount a GPU on demand, files survive when GPU is detached | A10G ~$0.80/h, A100 ~$2.50/h, free monthly credits | Lowest. Closest to "laptop with a bigger GPU". |
| **Modal** | Define a function in Python, decorator runs it on a GPU; files live in a Volume | A100 ~$3/h, billed per second | Lowest if you're comfortable with Python decorators; no SSH session to babysit. |
| **RunPod** | Rent a container with SSH/Jupyter, pay per hour, persistent network volume | A100 ~$2/h, 4090 ~$0.40/h | Low. Classic "rent a box". |
| **Lambda Labs** | Same shape as RunPod, slightly more polished | A100 ~$1.30/h | Low. |
| **vast.ai** | Cheapest by ~half, but it's a marketplace of random hosts | 4090 ~$0.25/h | Medium. Variable reliability. Fine for non-critical training. |
| **Colab Pro+** | Notebook-only, ~$50/mo, A100 sometimes | Flat sub | Lowest setup but session timeouts make multi-day runs painful. |

For this project: **Lightning AI Studios** if you want the laptop-like
experience, **RunPod with a 4090** if you want the cheapest hands-on box,
**Modal** if you want zero infrastructure to think about. A single 4090 is
plenty for a 30M-param model — you do not need an A100 here.

### 8b. The simplest possible workflow (RunPod + 4090)

```
1.  Pick the "PyTorch 2.5 / CUDA 12.4" template.
2.  Attach a 100 GB persistent network volume mounted at /workspace.
3.  Launch a 4090 pod (~$0.40/h).
4.  In the pod: clone the repo, generate the silver dataset once
    (saves to /workspace/data — survives pod restarts), train.
5.  Stop the pod when done. Volume keeps the checkpoint and dataset
    around at ~$0.05/GB/month. Restart later for fine-tuning.
```

End-to-end cost for the §7 milestone, on a 4090:

- Silver dataset generation (CPU-bound, 50k pairs, ~2 h): **$1**
- 3 epochs on 50k pairs with a 10M-param model: ~3 h, **$1.50**
- Gold-set fine-tune, 10 epochs: ~1 h, **$0.50**
- Storage for a month: **$5**

**Total under $10** to find out whether the approach works. Full 1M-pair
pretraining run: ~24 h on a 4090 ≈ **$10**, or ~6 h on an A100 ≈ **$15**.

### 8c. The even simpler workflow (Modal)

If you don't want to manage a box at all:

```python
# train.py — run with `modal run train.py`
import modal

image = modal.Image.debian_slim().pip_install(
    "torch==2.5.0", "transformers", "accelerate", "webdataset", "pillow"
)
volume = modal.Volume.from_name("teletext-data", create_if_missing=True)

app = modal.App("teletext-trainer")

@app.function(
    image=image,
    gpu="A10G",          # or "A100", "H100"
    volumes={"/data": volume},
    timeout=24 * 60 * 60,
)
def train():
    # ... your training loop, reads from /data, writes checkpoints to /data
    ...

@app.local_entrypoint()
def main():
    train.remote()
```

You run `modal run train.py` from the laptop; Modal spins up the GPU,
streams the logs back to your terminal, shuts the GPU down when the function
returns. Files in the Volume persist between runs. Billing is per-second so
a crashed run costs cents, not dollars. This is the lowest-overhead option
by some margin.

### 8d. Moving data and code in/out

- **Code**: `git push` from laptop, `git pull` in the pod. Don't try to
  rsync your working tree.
- **Datasets**: generate on the rented box, not locally — uploading 16 GB
  over a home connection is slow and pointless when the box has 1 Gbps.
  Source images: `aws s3 sync` from a public bucket (LAION, Open Images
  mirrors), or `huggingface-cli download` from an HF dataset. The gallery
  scrape can be re-run on the box in minutes.
- **Checkpoints**: write to the persistent volume during training; at the
  end, `huggingface-cli upload` the final checkpoint to a private HF repo.
  That gives you durable storage independent of any single provider, and
  pulling it back to the laptop for inference is a one-liner.
- **Secrets**: HF token, W&B key — set as environment variables/secrets in
  the provider's UI, never commit them.

### 8e. What to keep local

- The dataset-prep code and a tiny (1k-pair) shard for smoke tests.
- The OpenVINO-exported final model for inference and the Gradio UI.
- The eval harness that compares model output vs. DP solver output —
  rendering and scoring 50 pages is a CPU-seconds job and doesn't need a
  GPU.

The split is: **iterate locally on a tiny subset until the loop runs without
errors, then push and run the real job remotely**. Never debug on the rented
box — it bills by the second.

### 8f. Common pitfalls

- **Idle GPUs cost the same as busy ones.** Set a wall-clock timeout on
  every training script. RunPod and Lambda will happily bill you overnight
  if a job hangs on a dataloader. Modal auto-shuts-down on function return,
  which is why it's the safest option for unattended jobs.
- **Spot/interruptible instances** are ~50% cheaper but will be killed
  mid-training. Only use them if your training loop checkpoints every N
  steps and resumes cleanly. For a first attempt, pay the full rate.
- **Provider-specific PyTorch images often lag.** If you need PyTorch 2.5+
  features and the template ships 2.3, build your own image (Modal makes
  this trivial; on RunPod, `pip install --upgrade torch` in the pod and
  bake it into a custom template).

---

## 9. Milestone 1, step by step, on Modal

End-to-end recipe to find out whether a learned teletext model is viable, run
on Modal so the laptop only does git, editing, and the final eval. Budget:
**a few dollars and one evening of wall-clock time**.

The plan deliberately uses **one Modal app, one Volume, three functions**
(prep / train / sample). Keeping everything in one app means the same image
spec, the same secrets, and the same Volume mount across every step.

### Step 0 — Prerequisites (laptop, ~10 min)

```bash
pip install modal
modal setup            # opens a browser; logs you in and stores a token
modal secret create huggingface HF_TOKEN=hf_xxx   # for gated datasets, optional
```

Verify with `modal run -m modal.examples.hello_world` — should print "hello"
from a remote container.

### Step 1 — Repo and dataset prep code (laptop, ~30 min)

Add a `modal/` directory at the repo root with these files:

```
modal/
  app.py            # Modal app, image, volume, shared config
  prep_gold.py      # decode the 1900 gallery PNGs → byte streams
  prep_silver.py    # run image2teletext.py over a corpus
  train.py          # train the model
  sample.py         # generate sample outputs from a checkpoint
  model.py          # the VisionEncoderDecoder definition
```

`app.py` defines the shared bits exactly once:

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.0", "torchvision",
        "transformers==4.46", "accelerate", "datasets",
        "pillow", "numpy", "webdataset",
    )
    .add_local_dir(".", "/repo")        # bundles the working tree
)

volume = modal.Volume.from_name("teletext-m1", create_if_missing=True)
app    = modal.App("teletext-m1", image=image)

DATA = "/data"     # volume mount point used by every function
```

Each step below is one `@app.function(...)` in its own file that imports
`app`, `volume`, `DATA` from `app.py`.

### Step 2 — Generate the gold dataset (Modal, CPU, ~15 min, ~$0.10)

The 1900 gallery PNGs aren't on the rented box yet. Easiest: re-download them
*from* the box (your `download_gallery.py` already does this, with
rate-limiting). Then decode each to a 1000-byte stream with
`teletext_decode.py`.

```python
@app.function(volumes={DATA: volume}, timeout=60*60)
def prep_gold():
    import subprocess, sys, pathlib
    sys.path.insert(0, "/repo")
    # 1. Download (skips already-downloaded files thanks to your script)
    subprocess.run(["python", "/repo/gallery/download_gallery.py"], check=True)
    # 2. Decode each PNG → bytes, save as a single .npz of shape (N, 1000) uint8
    from teletext_decode import decode_image_to_bytes   # exposed by your module
    import numpy as np
    from PIL import Image
    paths = sorted(pathlib.Path("/repo/test/horsenburger").glob("*.png"))
    streams, kept = [], []
    for p in paths:
        try:
            b = decode_image_to_bytes(np.array(Image.open(p).convert("RGB")))
            assert len(b) == 1000
            streams.append(b); kept.append(p.name)
        except Exception:
            continue
    np.savez(f"{DATA}/gold.npz",
             bytes=np.array(streams, dtype=np.uint8),
             names=np.array(kept))
    volume.commit()
    print(f"gold pairs: {len(streams)}")
```

Run with `modal run modal/prep_gold.py::prep_gold`. The download step
re-runs each time you restart the function unless you check the volume first
— add a guard, or just let the script's existing `dest.exists()` skip handle
it (the files live under `/repo/test/horsenburger`, which is *inside the
container, not on the volume* — so the second run re-downloads. **Fix**: add
the gallery dir to the volume by pointing `OUT_DIR` at `/data/horsenburger`
via env var or a small wrapper).

### Step 3 — Generate the silver dataset (Modal, CPU, ~2 h, ~$1)

This is where the bulk of the data comes from. Use a small, easy-to-fetch
image corpus — the **`huggingface.co/datasets/sasha/dog-food`** or any
50k-image slice of LAION-aesthetic. Stream it, run `image2teletext.py` with
3 presets per image, write to a webdataset tar.

```python
@app.function(volumes={DATA: volume}, cpu=8, timeout=4*60*60)
def prep_silver(n_images: int = 50_000):
    import sys, io, tarfile
    sys.path.insert(0, "/repo")
    from datasets import load_dataset
    from image2teletext import convert      # expose as a function-level API
    from PIL import Image
    PRESETS = ["photo", "art", "vivid"]
    ds = load_dataset("...", split=f"train[:{n_images}]", streaming=True)
    with tarfile.open(f"{DATA}/silver.tar", "w") as tar:
        for i, row in enumerate(ds):
            img = row["image"].convert("RGB").resize((256, 192))
            for preset in PRESETS:
                bytes1000 = convert(img, preset=preset)
                key = f"{i:07d}_{preset}"
                # write image
                buf = io.BytesIO(); img.save(buf, "JPEG", quality=85)
                ti = tarfile.TarInfo(f"{key}.jpg"); ti.size = buf.tell()
                buf.seek(0); tar.addfile(ti, buf)
                # write bytes
                ti = tarfile.TarInfo(f"{key}.bin"); ti.size = 1000
                tar.addfile(ti, io.BytesIO(bytes1000))
            if i % 1000 == 0: print(f"{i}/{n_images}"); volume.commit()
    volume.commit()
```

Run with `modal run modal/prep_silver.py::prep_silver --n-images 50000`.

The `convert(image, preset=..., **overrides)` API used above is already
exposed at the top level of `image2teletext.py`. It accepts a path, a PIL
Image, or an HxWx3 uint8 numpy array, and returns a 1000-byte `bytes`
object — no CLI subprocess needed.

### Step 4 — Define the model (laptop, ~1 h)

`model.py`, ~80 lines. Use HuggingFace's `VisionEncoderDecoderModel` to
avoid writing the cross-attention plumbing yourself:

```python
from transformers import (
    VisionEncoderDecoderModel, ViTConfig, ViTModel,
    GPT2Config, GPT2LMHeadModel,
)

VOCAB = 260   # 256 bytes + <bos> <eos> <pad> <unused>

def build_model():
    enc = ViTModel(ViTConfig(
        image_size=(192, 256), patch_size=16,
        hidden_size=192, num_hidden_layers=6, num_attention_heads=4,
    ))
    dec = GPT2LMHeadModel(GPT2Config(
        vocab_size=VOCAB, n_positions=1024,
        n_embd=256, n_layer=8, n_head=4,
        bos_token_id=256, eos_token_id=257, pad_token_id=258,
        add_cross_attention=True,
    ))
    m = VisionEncoderDecoderModel(encoder=enc, decoder=dec)
    m.config.decoder_start_token_id = 256
    m.config.pad_token_id           = 258
    return m   # ~10–12 M params
```

Smoke test locally on CPU with batch 2 to confirm forward+backward run.

### Step 5 — Train (Modal, single A10G, ~3 h, ~$2)

```python
@app.function(
    volumes={DATA: volume},
    gpu="A10G",                # bump to A100 only if A10G is too slow
    timeout=6*60*60,
    secrets=[modal.Secret.from_name("wandb")],   # optional
)
def train(epochs: int = 3, batch_size: int = 32, lr: float = 3e-4):
    import sys; sys.path.insert(0, "/repo")
    import torch, webdataset as wds
    from modal.model import build_model
    # ...standard training loop:
    #   - webdataset pipeline reading /data/silver.tar
    #   - AdamW, cosine LR, bf16 autocast, grad clip 1.0
    #   - cross-entropy on the 1000 target bytes (+ start token)
    #   - checkpoint to /data/ckpt-{epoch}.pt every epoch
    #   - log every 50 steps
    ...
    volume.commit()
```

Run with `modal run modal/train.py::train --epochs 3`. Watch the live logs in
the terminal. Modal shuts the GPU down the second the function returns, so
you cannot accidentally leave it billing.

A 10M-param model on 50k × 3 = 150k samples × 3 epochs ≈ 450k steps at
batch 32 ≈ 14k optimizer steps. On an A10G that's roughly **2–3 hours**.

### Step 6 — Sample and evaluate (Modal CPU + laptop, ~10 min, ~$0.05)

Two pieces:

1. On Modal, generate predictions for a held-out set of (say) 100 gold images
   and write `samples.npz` (predicted byte streams) to the volume.
2. On the laptop, `modal volume get teletext-m1 samples.npz`, then run a
   local script that:
    - renders each predicted byte stream via `teletext_decode.py` (its
      reverse, byte → PNG)
    - computes byte accuracy vs. the gold
    - computes stream-validity rate (does it decode without error?)
    - dumps a side-by-side HTML grid you can open in the browser

```python
@app.function(volumes={DATA: volume}, gpu="A10G", timeout=30*60)
def sample(ckpt: str = "ckpt-3.pt", n: int = 100):
    # ...load model, load 100 gold images, generate, save to /data/samples.npz
    ...
```

Run: `modal run modal/sample.py::sample --ckpt ckpt-3.pt`, then locally
`modal volume get teletext-m1 samples.npz ./` and run your eval HTML
generator.

### Step 7 — Decide

The decision criteria from §7:

- **Stream validity ≥ 95%** → architecture works; invest in the cloud
  pretraining run on the 1M silver set + gold fine-tune.
- **Stream validity 50–95%** → AR is right but the decoder is undertrained
  or too small. Bump decoder layers 8 → 12, train 3 → 10 epochs, retry. One
  more $5 run.
- **Stream validity <50%** → either data prep is broken (likely) or the
  framing is wrong. Check that `convert()` outputs round-trip through
  `teletext_decode.py` byte-identical first.

### Progress log

- **Step 0 (Modal setup)** — done. Client `1.4.2` installed, authenticated to
  the `kieranhj` workspace, token at `~/.modal.toml`.
- **Step 1 (shared image + volume)** — done. `modal_jobs/verify.py` runs
  end-to-end: image cached (`torch 2.5.1+cu124`, `transformers 4.46.3`),
  `teletext-m1` Volume created and round-tripped, `image2teletext` /
  `teletext_decode` import cleanly inside the container. App URL pattern:
  `https://modal.com/apps/kieranhj/main/<run-id>`.

  Convention changes from the original §9 plan that future steps must
  inherit (see `memory/reference_modal_setup.md` for full rationale):
    - Job dir is `modal_jobs/`, not `modal/` (avoids shadowing the modal
      pip package).
    - Each script **inlines** its own `image` / `volume` / `app` definition;
      no shared `app.py`. Modal re-imports the entrypoint inside the
      container, so cross-script imports break server-side.
    - Image mounts list specific files, never `add_local_dir(REPO_ROOT)` —
      the repo is ~700 MB and the user has ~3 Mb/s upload.
    - Run via `./modal.bat run modal_jobs/<script>.py` from the repo root
      (the wrapper sets `PYTHONUTF8=1` so the CLI's Unicode output
      doesn't crash the Windows console).
    - Cast non-trivial return values to plain types before returning
      (e.g. `str(torch.__version__)`) so the laptop can unpickle them
      without needing the same packages installed.

### Next steps (resume here)

- **Step 2 — `prep_gold`**. New file `modal_jobs/prep_gold.py`. Inline the
  same image+volume boilerplate as `verify.py`. Function should:
    1. Set `GALLERY_OUT_DIR=/data/horsenburger` so
       `download_gallery.py` writes to the volume (already wired up in
       `gallery/download_gallery.py`).
    2. Run the downloader (it skips already-present files, so re-runs are
       cheap).
    3. Iterate the PNGs in `/data/horsenburger`, decode each via
       `teletext_decode.decode_image_to_bytes`, write
       `/data/gold.npz` with arrays `bytes` (N×1000 uint8) and `names`
       (N strings).
    4. `volume.commit()` and return counts.
  Expected: ~15 min wall, ~$0.10. All bandwidth is modal→horsenburger,
  none from the laptop.
- **Step 3 — `prep_silver`**. Then training, sampling, eval as in §9.

### Total budget summary

| Step | Resource | Wall time | Cost |
|---|---|---|---|
| 2. Gold prep | CPU | 15 min | $0.10 |
| 3. Silver prep | 8 vCPU | 2 h | $1.00 |
| 5. Train | A10G | 3 h | $2.50 |
| 6. Sample | A10G | 10 min | $0.20 |
| Volume storage (1 month) | — | — | $0.50 |
| **Total** | | **~6 h wall, evening** | **~$5** |

If steps 2/3 turn out to be slow because they're CPU-bound and your code
isn't multiprocess, parallelise across Modal containers with
`.map()` — Modal will fan out to dozens of workers and the cost stays
identical (you pay per CPU-second, not per machine).

## 10. M1 results and M2 attempts (2026-05-14)

### M1 outcome — success

12M-param ViT-tiny + GPT2-small VisionEncoderDecoder, trained on 150k silver
pairs (50k wikiart × 3 presets) for 3 epochs on a single A10G.

- Wall: ~50 min training + ~8 min silver prep + ~5 min gold prep
- Cost: ~$1.50 total
- Final ckpt: `/data/ckpt-03.pt` on `teletext-m1` volume (avg loss 0.699)
- Visual quality on gold horsenburger eval set: clearly conditions on input,
  picks up colours and coarse shapes well, learned subtle HOLD-graphics
  behaviour, surprisingly few control-code errors. Some weakness on fine
  detail and on out-of-distribution pages (e.g. pure B&W).
- Byte-match against gold: misleading metric. Loss halved 1.5 → 0.7 from
  epoch 1 → 3 while byte-match *dropped* 424 → 373/1000. Trust visual /
  re-rendered-pixel / stream-validity instead. See
  `memory/project_eval_metrics.md`.

### M2 attempts — both failed, bronze is poison

We have ~112k real teletext pages on disk (`datasets/computer_legacy/pages/`
+ `datasets/teletextart/pages/`), each a 1000-byte stream. The plan was to
render them back to PNGs and use `(render(bytes), bytes)` as bronze training
pairs to fix M1's residual control-code errors.

Tried two strategies, both regressed:

- **M2 — 30/70 bronze/silver mix from scratch, 3 epochs**: outputs garbled,
  two near-blank, only one image improved.
- **M2b — bronze-only fine-tune from M1 ckpt-03, 1 epoch, lr 3e-5**:
  catastrophic forgetting. All outputs garbage, no image adherence, even
  the always-easy idx-7 test image dropped from 908 to 77 byte-match.

**Root cause** (don't try bronze again without addressing this): in bronze,
`image = render(bytes)` makes the task near-identity. It's much easier than
photo→bytes, so the model preferentially learns it and abandons the hard
task. Bronze isn't extra grammar examples — it's a different, easier task
that crowds out the real one. Lesson saved in
`memory/project_eval_metrics.md`.

### Plan for next session

M1 ckpt-03 stands as the project's best model. The remaining headroom is
in things that don't change the underlying training task:

#### Ranked options (cheapest-first)

1. **Improve silver targets via a better DP solver** *(pure CPU, ~hours)*.
   The HOLD-graphics errors we wanted bronze to fix are also present in
   the silver targets (the DP solver in `image2teletext.py` doesn't always
   pick the cleanest control-code placement). Fixing the solver lets us
   regenerate silver and retrain at no extra GPU cost beyond one more
   3-epoch run (~$1). Highest leverage per dollar.
   - Concrete: read `image2teletext.py`, find the DP solver, audit how it
     handles HOLD-GFX / SET-AT transitions, add a cell-cost term that
     penalises gaps and background bleed.
   - Validation: rerun on gold images, compare new vs old silver outputs
     visually before retraining.

2. **Add gold to training** *(near-zero new compute)*. Right now all 1909
   horsenburger gold pages are held out for eval. Split 80/20: ~1500
   training pairs of real photo→DP-solver-bytes that don't go through any
   render-roundtrip. Mix into silver at ~5% (so they get oversampled
   relative to silver size). Cheap to try — same architecture, same loop,
   just an extra shard or two.
   - Risk: only 1500 pairs, may not move the needle.

3. **Scale silver to 200k images** *(~30 min CPU prep, $1 GPU)*. Same
   wikiart streaming, longer run. Uses the existing
   `prep_silver.py` skip-resume to extend the dataset incrementally. Gives
   the model more photo diversity, especially the rare modes (B&W,
   monochrome, high-contrast) that wikiart-50k undersamples.

4. **Bigger model — ViT-small + GPT2-medium** *(~50M params, ~3 h GPU,
   ~$3)*. Test if 12M is the bottleneck before throwing more data at it.
   Worth doing only after (1) and (3) are established as not-enough.

#### Off the table

- **Bronze / `(render(bytes), bytes)` pairs in any form**, until we have
  a way to make the encoder genuinely solve a different problem on bronze
  vs silver (e.g. degrade bronze inputs with noise / down-sampling, OR
  use bronze only for decoder-LM-style pretraining with image inputs
  zeroed out). Don't repeat the naive approach.

#### Resume-here checklist

- M1 ckpt is at `/data/ckpt-03.pt` on the `teletext-m1` Modal volume.
- Untouched gold dataset at `/data/gold.npz` (1909 pairs).
- Silver shards at `/data/silver/silver-*.tar` (50 shards, 150k pairs).
- Bronze shards exist at `/data/bronze/bronze-*.tar` but **don't use
  them** unless first solving the identity-task issue above.
- All M2 ckpts (`ckpt-m2-*.pt`, `ckpt-m2b-*.pt`) can be deleted to free
  volume space.
- `modal_jobs/train.py` already supports `--resume-ckpt`, `--bronze-frac`,
  `--tag`, `--warmup-steps`, `--lr`. The `MSYS_NO_PATHCONV=1` env var is
  required when passing `/data/...` paths from Git Bash.
