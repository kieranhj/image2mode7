"""Step 5 — train.

Train the VisionEncoderDecoder model on the silver shards.

Pipeline:
  webdataset stream /data/silver/silver-*.tar
    -> decode .jpg -> (3, 192, 256) float
    -> decode .bin -> (1001,) long  (BOS + 1000 byte targets)
    -> AdamW + cosine LR + bf16 autocast + grad clip 1.0
    -> checkpoint /data/ckpt-{epoch}.pt every epoch

Run:
  ./modal.bat run --detach modal_jobs/train.py --spawn
"""
import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_FILES = [
    "image2teletext.py",
    "teletext_decode.py",
]
MODAL_FILES = [
    "modal_jobs/model.py",
]

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "torchvision==0.20.1",
    "transformers==4.46.3", "accelerate==1.1.1",
    "webdataset==0.2.100",
    "pillow==11.0.0", "numpy==2.1.3", "tqdm",
)
for _f in REPO_FILES:
    image = image.add_local_file(str(REPO_ROOT / _f), f"/repo/{_f}")
for _f in MODAL_FILES:
    image = image.add_local_file(str(REPO_ROOT / _f),
                                 f"/repo/{Path(_f).name}")

volume = modal.Volume.from_name("teletext-m1", create_if_missing=True)
app    = modal.App("teletext-m1-train", image=image)
DATA   = "/data"

SILVER_GLOB = f"{DATA}/silver/silver-*.tar"
BRONZE_GLOB = f"{DATA}/bronze/bronze-*.tar"
CKPT_FMT    = f"{DATA}/ckpt-{{tag}}-{{epoch:02d}}.pt"
LOG_EVERY   = 50


@app.function(volumes={DATA: volume}, gpu="A10G", timeout=6 * 60 * 60)
def train(epochs: int = 3,
          batch_size: int = 32,
          lr: float = 3e-4,
          weight_decay: float = 0.05,
          warmup_steps: int = 500,
          num_workers: int = 4,
          bronze_frac: float = 0.0,
          tag: str = "m1",
          resume_ckpt: str = ""):
    import sys, glob, math, time, os, io
    sys.path.insert(0, "/repo")
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    import webdataset as wds
    from torchvision import transforms
    from model import (build_model, PAGE_LEN, BOS_ID, EOS_ID, PAD_ID, IMAGE_HW)

    volume.reload()
    silver_shards = sorted(glob.glob(SILVER_GLOB)) if bronze_frac < 1.0 else []
    bronze_shards = sorted(glob.glob(BRONZE_GLOB)) if bronze_frac > 0 else []
    if bronze_frac < 1.0 and not silver_shards:
        raise RuntimeError(f"no silver shards at {SILVER_GLOB}")
    if bronze_frac > 0 and not bronze_shards:
        raise RuntimeError(f"bronze_frac>0 but no bronze shards at {BRONZE_GLOB}")
    print(f"[train] silver={len(silver_shards)} bronze={len(bronze_shards)} "
          f"bronze_frac={bronze_frac} batch={batch_size} lr={lr} "
          f"epochs={epochs} tag={tag}", flush=True)

    device = torch.device("cuda")
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M", flush=True)

    if resume_ckpt:
        print(f"[train] loading weights from {resume_ckpt}", flush=True)
        ck = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                              betas=(0.9, 0.95))

    img_tf = transforms.Compose([
        transforms.ToTensor(),  # uint8 HWC -> float [0,1] CHW
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    def decode_sample(sample):
        from PIL import Image
        img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
        if img.size != (IMAGE_HW[1], IMAGE_HW[0]):
            img = img.resize((IMAGE_HW[1], IMAGE_HW[0]), Image.LANCZOS)
        px = img_tf(img)
        page = torch.frombuffer(bytearray(sample["bin"]), dtype=torch.uint8).long()
        assert page.numel() == PAGE_LEN
        decoder_input_ids = torch.cat([torch.tensor([BOS_ID]), page])
        labels            = torch.cat([page, torch.tensor([EOS_ID])])
        return px, decoder_input_ids, labels

    def make_one(shards, seed):
        return (wds.WebDataset(shards, shardshuffle=True,
                               nodesplitter=wds.split_by_node,
                               seed=seed, handler=wds.warn_and_continue)
                .shuffle(1000)
                .map(decode_sample, handler=wds.warn_and_continue))

    def make_loader(epoch_seed):
        if bronze_frac >= 1.0:
            ds = make_one(bronze_shards, epoch_seed).batched(
                batch_size, partial=False)
        elif bronze_frac > 0:
            silver_ds = make_one(silver_shards, epoch_seed)
            bronze_ds = make_one(bronze_shards, epoch_seed + 1)
            mix = wds.RandomMix([silver_ds, bronze_ds],
                                probs=[1.0 - bronze_frac, bronze_frac])
            ds = wds.DataPipeline(mix, wds.batched(batch_size, partial=False))
        else:
            ds = make_one(silver_shards, epoch_seed).batched(
                batch_size, partial=False)
        return DataLoader(ds, batch_size=None, num_workers=num_workers,
                          pin_memory=True, persistent_workers=False)

    # Rough sample count for cosine schedule. Silver = 3 samples/source*image (3
    # presets), bronze = 1. With RandomMix, an epoch is bounded by whichever
    # stream we'd exhaust first at the chosen mix ratio.
    silver_samples = len(silver_shards) * 1000 * 3
    bronze_samples = len(bronze_shards) * 1000
    if bronze_frac >= 1.0:
        total_samples = bronze_samples
    elif bronze_frac > 0:
        total_samples = int(min(silver_samples / (1.0 - bronze_frac),
                                bronze_samples / bronze_frac))
    else:
        total_samples = silver_samples
    steps_per_epoch = total_samples // batch_size
    total_steps     = steps_per_epoch * epochs
    print(f"[train] est total steps: {total_steps} "
          f"({steps_per_epoch}/epoch)", flush=True)

    def lr_at(step):
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scaler_dtype = torch.bfloat16
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        loader = make_loader(epoch_seed=epoch)
        ep_loss_sum, ep_loss_n = 0.0, 0

        for batch in loader:
            px, dec_in, labels = batch
            px       = px.to(device, non_blocking=True)
            dec_in   = dec_in.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            for g in optim.param_groups:
                g["lr"] = lr_at(step)

            with torch.autocast(device_type="cuda", dtype=scaler_dtype):
                out = model(pixel_values=px, decoder_input_ids=dec_in,
                            labels=labels)
                loss = out.loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            ep_loss_sum += float(loss.item())
            ep_loss_n   += 1
            step += 1

            if step % LOG_EVERY == 0:
                dt = time.time() - t0
                spd = step / dt
                print(f"  ep={epoch} step={step}/{total_steps} "
                      f"lr={lr_at(step):.2e} loss={loss.item():.3f} "
                      f"{spd:.1f} step/s", flush=True)

        ckpt_path = CKPT_FMT.format(tag=tag, epoch=epoch + 1)
        torch.save({
            "model": model.state_dict(),
            "config": model.config.to_diff_dict(),
            "epoch": epoch + 1,
            "step": step,
        }, ckpt_path)
        volume.commit()
        avg = ep_loss_sum / max(1, ep_loss_n)
        print(f"[train] epoch {epoch+1} done — avg_loss={avg:.3f}, "
              f"ckpt -> {ckpt_path}", flush=True)

    return {
        "epochs": epochs,
        "steps": step,
        "final_ckpt": CKPT_FMT.format(tag=tag, epoch=epochs),
        "wall_minutes": round((time.time() - t0) / 60, 1),
    }


@app.local_entrypoint()
def main(epochs: int = 3, batch_size: int = 32, lr: float = 3e-4,
         warmup_steps: int = 500, bronze_frac: float = 0.0, tag: str = "m1",
         resume_ckpt: str = "", spawn: bool = False):
    args = (epochs, batch_size, lr, 0.05, warmup_steps, 4,
            bronze_frac, tag, resume_ckpt)
    if spawn:
        call = train.spawn(*args)
        print(f"spawned: {call.object_id}")
    else:
        print(train.remote(*args))
