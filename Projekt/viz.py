"""Erzeugt Abbildungen fuer Praesentation/Doku:
  assets/samples.png      -- Beispielbilder des synthetischen Datensatzes
  assets/similarity.png   -- Bild<->Text-Aehnlichkeitsmatrix (gelerntes Alignment)
"""
import os

import numpy as np
import torch
from PIL import Image

from config import Config
from src.data import SyntheticShapes, Collate
from src.model import MiniCLIP
from src.tokenizer import SimpleTokenizer

os.makedirs("assets", exist_ok=True)


def denorm(t):
    arr = (t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
    return arr.clip(0, 255).astype(np.uint8)


def make_samples(n=8, size=64):
    ds = SyntheticShapes(n, size, seed=7)
    pad = 4
    grid = Image.new("RGB", (n * (size + pad) + pad, size + 2 * pad), (255, 255, 255))
    for i in range(n):
        item = ds[i]
        grid.paste(Image.fromarray(denorm(item["image"])), (pad + i * (size + pad), pad))
    grid.save("assets/samples.png")
    print("assets/samples.png", [SyntheticShapes(n, size, seed=7)[i]["caption"] for i in range(n)])


def make_similarity(ckpt="checkpoints/mini_clip.pt", k=8):
    if not os.path.exists(ckpt):
        print("kein Checkpoint -> ueberspringe similarity.png"); return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib fehlt -> ueberspringe similarity.png"); return

    cfg = Config()
    state = torch.load(ckpt, map_location="cpu")
    tok = SimpleTokenizer().load_state_dict(state["tokenizer"])
    model = MiniCLIP(cfg, tok.vocab_size, tok.pad_id); model.load_state_dict(state["model"]); model.eval()
    collate = Collate(tok)

    # k Bilder mit moeglichst unterschiedlichen Captions waehlen
    ds = SyntheticShapes(200, cfg.image_size, seed=3)
    seen, picks = set(), []
    for i in range(len(ds)):
        it = ds[i]
        if it["caption"] not in seen:
            seen.add(it["caption"]); picks.append(it)
        if len(picks) == k: break

    batch = collate(picks)
    with torch.no_grad():
        img = model.encode_image(batch["image"]); txt = model.encode_text(batch["text"])
        sim = (img @ txt.t()).numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([p["caption"].replace("a photo of ", "") for p in picks],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"Bild {i}" for i in range(k)], fontsize=8)
    ax.set_title("Bild-Text-Ähnlichkeit (Diagonale = korrekte Paare)")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    color="white" if sim[i, j] < sim.max() * 0.6 else "black", fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig("assets/similarity.png", dpi=130)
    print("assets/similarity.png")


if __name__ == "__main__":
    make_samples()
    make_similarity()
