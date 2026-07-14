"""Evaluation: Bild<->Text-Retrieval (Recall@K) und Zero-Shot-Klassifikation."""
from typing import List

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def encode_dataset(model, dataset, collate, batch_size=128, device="cpu"):
    """Gibt L2-normalisierte Bild- und Text-Embeddings + Labels zurueck."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate)
    img_emb, txt_emb, labels = [], [], []
    for batch in loader:
        img_emb.append(model.encode_image(batch["image"].to(device)).cpu())
        txt_emb.append(model.encode_text(batch["text"].to(device)).cpu())
        labels.append(batch["label"])
    return torch.cat(img_emb), torch.cat(txt_emb), torch.cat(labels)


@torch.no_grad()
def retrieval_recall(img_emb, txt_emb, ks=(1, 5, 10)):
    """Recall@K fuer beide Richtungen. Annahme: img_emb[i] gehoert zu txt_emb[i].

    Hinweis: Bei Flickr8k teilen sich mehrere Captions ein Bild. Wir werten hier
    bewusst paarweise (1:1) aus -- das ist eine strenge, aber faire Metrik, die
    fuer Mini-CLIP und die Baseline identisch berechnet wird.
    """
    sim = img_emb @ txt_emb.t()                 # [N, N]
    n = sim.size(0)
    gt = torch.arange(n)
    out = {}
    # Bild -> Text
    ranks_i = sim.argsort(dim=1, descending=True)
    # Text -> Bild
    ranks_t = sim.t().argsort(dim=1, descending=True)
    for k in ks:
        hit_i = (ranks_i[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item()
        hit_t = (ranks_t[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item()
        out[f"i2t_R@{k}"] = round(100 * hit_i, 2)
        out[f"t2i_R@{k}"] = round(100 * hit_t, 2)
    return out


@torch.no_grad()
def zero_shot_classify(model, dataset, collate, classnames: List[str],
                       prompt="a photo of a {}", batch_size=128, device="cpu"):
    """Klassifiziert Bilder ohne weiteres Training, nur ueber Text-Prompts der
    Klassennamen -- der zentrale Trick aus dem CLIP-Paper."""
    model.eval()
    tok = collate.tok
    class_tokens = tok.encode_batch([prompt.format(c) for c in classnames]).to(device)
    class_emb = model.encode_text(class_tokens)         # [num_classes, d]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate)
    correct, total = 0, 0
    for batch in loader:
        labels = batch["label"]
        if (labels < 0).any():       # Datensatz ohne Klassenlabels (z.B. Flickr)
            return None
        img = model.encode_image(batch["image"].to(device))
        preds = (img @ class_emb.t()).argmax(1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return round(100 * correct / total, 2)


def pretty(metrics: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in metrics.items())


if __name__ == "__main__":
    # Eigenstaendige Auswertung eines gespeicherten Checkpoints auf dem Synthetik-Set.
    import argparse
    from config import Config
    from src.data import SyntheticShapes, Collate
    from src.model import MiniCLIP
    from src.tokenizer import SimpleTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mini_clip.pt")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    cfg = Config()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    tok = SimpleTokenizer().load_state_dict(ckpt["tokenizer"])
    model = MiniCLIP(cfg, tok.vocab_size, tok.pad_id)
    model.load_state_dict(ckpt["model"])
    collate = Collate(tok)

    ds = SyntheticShapes(n_samples=args.n, image_size=cfg.image_size, seed=999)
    img_emb, txt_emb, _ = encode_dataset(model, ds, collate, device="cpu")
    print("Retrieval:", pretty(retrieval_recall(img_emb, txt_emb)))
    acc = zero_shot_classify(model, ds, collate, SyntheticShapes.classnames)
    print("Zero-Shot-Accuracy:", acc, "%")
