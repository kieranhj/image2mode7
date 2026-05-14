"""Step 4 — model definition.

VisionEncoderDecoderModel = ViT-tiny encoder + GPT2-small decoder + cross-attn.
Input: (B, 3, 192, 256) image. Target: (B, 1001) tokens (bos + 1000 byte ids).

Vocab is 260: 0..255 are the 256 possible page bytes, plus <bos>=256, <eos>=257,
<pad>=258, <unused>=259. We don't strictly need <eos> (page length is fixed at
1000) but include it for symmetry with HF training utilities.
"""
from transformers import (
    VisionEncoderDecoderModel, ViTConfig, ViTModel,
    GPT2Config, GPT2LMHeadModel,
)

VOCAB    = 260
BOS_ID   = 256
EOS_ID   = 257
PAD_ID   = 258

IMAGE_HW = (192, 256)
PAGE_LEN = 1000


def build_model():
    """Builds the image -> 1000-byte teletext page model. ~10-12M params."""
    enc = ViTModel(ViTConfig(
        image_size=IMAGE_HW,
        patch_size=16,
        hidden_size=192,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=768,
        num_channels=3,
    ))
    dec = GPT2LMHeadModel(GPT2Config(
        vocab_size=VOCAB,
        n_positions=PAGE_LEN + 8,
        n_embd=256,
        n_layer=8,
        n_head=4,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        add_cross_attention=True,
    ))
    model = VisionEncoderDecoderModel(encoder=enc, decoder=dec)
    model.config.decoder_start_token_id = BOS_ID
    model.config.pad_token_id           = PAD_ID
    model.config.eos_token_id           = EOS_ID
    return model


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import torch
    m = build_model()
    print(f"params: {n_params(m)/1e6:.2f}M")

    B = 2
    px  = torch.randn(B, 3, *IMAGE_HW)
    tgt = torch.randint(0, 256, (B, PAGE_LEN))
    bos = torch.full((B, 1), BOS_ID, dtype=torch.long)
    labels = torch.cat([tgt, torch.full((B, 1), EOS_ID, dtype=torch.long)], dim=1)
    decoder_input_ids = torch.cat([bos, tgt], dim=1)

    out = m(pixel_values=px, decoder_input_ids=decoder_input_ids, labels=labels)
    print(f"logits: {tuple(out.logits.shape)}  loss: {out.loss.item():.3f}")
