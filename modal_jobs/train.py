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

SHARDS_GLOB = f"{DATA}/silver/silver-*.tar"
CKPT_FMT    = f"{DATA}/ckpt-{{:02d}}.pt"
LOG_EVERY   = 50


@app.function(volumes={DATA: volume}, gpu="A10G", timeout=6 * 60 * 60)
def train(epochs: int = 3,
          batch_size: int = 32,
          lr: float = 3e-4,
          weight_decay: float = 0.05,
          warmup_steps: int = 500,
          num_workers: int = 4):
    import sys, glob, math, time, os, io
    sys.path.insert(0, "/repo")
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    import webdataset as wds
    from torchvision import transforms
    from model import (build_model, PAGE_LEN, BOS_ID, EOS_ID, PAD_ID, IMAGE_HW)

    volume.reload()
    shards = sorted(glob.glob(SHARDS_GLOB))
    if not shards:
        raise RuntimeError(f"no shards found at {SHARDS_GLOB}")
    print(f"[train] {len(shards)} shards, batch={batch_size}, lr={lr}, "
          f"epochs={epochs}", flush=True)

    device = torch.device("cuda")
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M", flush=True)

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

    def make_loader(epoch_seed):
        ds = (wds.WebDataset(shards, shardshuffle=True, nodesplitter=wds.split_by_node,
                             seed=epoch_seed, handler=wds.warn_and_continue)
              .shuffle(2000)
              .map(decode_sample, handler=wds.warn_and_continue)
              .batched(batch_size, partial=False))
        return DataLoader(ds, batch_size=None, num_workers=num_workers,
                          pin_memory=True, persistent_workers=False)

    # rough sample count for cosine schedule
    samples_per_shard = 1000 * 3
    total_samples = len(shards) * samples_per_shard
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

        ckpt_path = CKPT_FMT.format(epoch + 1)
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
        "final_ckpt": CKPT_FMT.format(epochs),
        "wall_minutes": round((time.time() - t0) / 60, 1),
    }


@app.local_entrypoint()
def main(epochs: int = 3, batch_size: int = 32, lr: float = 3e-4,
         spawn: bool = False):
    if spawn:
        call = train.spawn(epochs, batch_size, lr)
        print(f"spawned: {call.object_id}")
    else:
        print(train.remote(epochs, batch_size, lr))
