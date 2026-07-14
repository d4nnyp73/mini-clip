"""Baseline: vortrainiertes CLIP (open_clip) auf demselben Test-Split.

Zweck: einordnen, wie weit unser winziges, auf wenigen tausend Paaren
trainiertes Mini-CLIP von einem auf 400M Paaren vortrainierten Original
entfernt ist. laeuft auch auf CPU (einige Minuten).

Installation der Baseline (optional):
    pip install open_clip_torch
"""
import argparse

import torch
from torch.utils.data import DataLoader

from config import Config
from src.data import SyntheticShapes, Flickr8kDataset
from src.evaluate import retrieval_recall, pretty


def load_open_clip():
    try:
        import open_clip
    except ImportError:
        raise SystemExit(
            "open_clip nicht installiert. -> pip install open_clip_torch")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def evaluate(dataset, n_max=500):
    """Erwartet ein Dataset, das PIL-faehige Bilder rekonstruieren kann. Wir
    nutzen hier die schon vorhandenen Tensoren und de-normalisieren sie zurueck
    zu PIL fuer den open_clip-Preprocessor."""
    import numpy as np
    from PIL import Image

    model, preprocess, tokenizer = load_open_clip()
    n = min(len(dataset), n_max)

    imgs, txts = [], []
    for i in range(n):
        item = dataset[i]
        arr = ((item["image"].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255)
        pil = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        imgs.append(preprocess(pil))
        txts.append(item["caption"])

    image_input = torch.stack(imgs)
    text_input = tokenizer(txts)
    img_emb = torch.nn.functional.normalize(model.encode_image(image_input), dim=-1)
    txt_emb = torch.nn.functional.normalize(model.encode_text(text_input), dim=-1)
    return retrieval_recall(img_emb.float(), txt_emb.float())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "flickr8k"], default="synthetic")
    ap.add_argument("--data-root", default="data/flickr8k")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    cfg = Config()
    if args.dataset == "synthetic":
        ds = SyntheticShapes(args.n, cfg.image_size, seed=999)
    else:
        ds = Flickr8kDataset(args.data_root, cfg.image_size, split="val")

    print("Pretrained CLIP (ViT-B/32) Baseline:")
    print("  " + pretty(evaluate(ds, args.n)))
