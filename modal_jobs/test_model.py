"""Smoke-test modal_jobs/model.py on Modal (we have no local torch/transformers)."""
import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "torchvision==0.20.1",
    "transformers==4.46.3", "accelerate==1.1.1",
    "numpy==2.1.3",
)
image = image.add_local_file(
    str(REPO_ROOT / "modal_jobs" / "model.py"), "/repo/model.py")

app = modal.App("teletext-m1-test", image=image)


@app.function(timeout=300)
def test_build():
    import sys, torch
    sys.path.insert(0, "/repo")
    from model import build_model, n_params, IMAGE_HW, PAGE_LEN, BOS_ID, EOS_ID

    m = build_model()
    n = n_params(m)

    B = 2
    px  = torch.randn(B, 3, *IMAGE_HW)
    tgt = torch.randint(0, 256, (B, PAGE_LEN))
    bos = torch.full((B, 1), BOS_ID, dtype=torch.long)
    labels = torch.cat([tgt, torch.full((B, 1), EOS_ID, dtype=torch.long)], dim=1)
    decoder_input_ids = torch.cat([bos, tgt], dim=1)

    out = m(pixel_values=px, decoder_input_ids=decoder_input_ids, labels=labels)
    return {
        "params_M": round(n / 1e6, 2),
        "logits_shape": list(out.logits.shape),
        "loss": float(out.loss.item()),
    }


@app.local_entrypoint()
def main():
    print(test_build.remote())
