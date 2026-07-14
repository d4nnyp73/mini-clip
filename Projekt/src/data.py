"""Datensaetze fuer Mini-CLIP.

Zwei Stufen:
  1) SyntheticShapes  -- selbst erzeugte (Bild, Caption)-Paare. Braucht keinen
     Download, laeuft in Sekunden und dient als Sanity-Check: das Modell MUSS
     hier nahezu perfektes Retrieval lernen, sonst stimmt etwas mit der
     Implementierung nicht. Hat zugleich saubere Klassen -> Zero-Shot-Benchmark.
  2) Flickr8kDataset  -- echte Bild-Text-Daten (8000 Bilder, je 5 Captions).
     Naechster, kleiner Verwandter der CLIP-Trainingsdaten.

Beide liefern Dicts {"image": FloatTensor[3,H,W], "caption": str, "label": int}.
"""
import json
import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

# ImageNet-Statistik wird hier nicht gebraucht; wir normalisieren auf [-1, 1].
_MEAN, _STD = 0.5, 0.5


def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, [0,1]
    arr = (arr - _MEAN) / _STD                                # [-1,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW


# --------------------------------------------------------------------------- #
#  1) Synthetischer Datensatz
# --------------------------------------------------------------------------- #
_COLORS = {
    "red": (220, 40, 40), "green": (40, 180, 70), "blue": (50, 90, 220),
    "yellow": (240, 215, 40), "purple": (150, 50, 200),
}
_SHAPES = ["circle", "square", "triangle", "star"]


class SyntheticShapes(Dataset):
    """Bunte Formen auf grauem Grund. Caption = 'a photo of a {color} {shape}'.
    label = Index der Form (fuer Zero-Shot-Klassifikation)."""

    classnames = _SHAPES

    def __init__(self, n_samples: int = 2000, image_size: int = 64, seed: int = 0):
        self.size = image_size
        rng = random.Random(seed)
        self.items = []
        for _ in range(n_samples):
            shape_idx = rng.randrange(len(_SHAPES))
            shape = _SHAPES[shape_idx]
            color = rng.choice(list(_COLORS))
            self.items.append((shape_idx, shape, color, rng.randint(0, 10_000)))

    def _render(self, shape: str, color: str, jitter: int) -> Image.Image:
        s = 96  # in hoher Aufloesung zeichnen, danach verkleinern
        img = Image.new("RGB", (s, s), (110, 110, 110))
        d = ImageDraw.Draw(img)
        r = random.Random(jitter)
        pad = r.randint(8, 20)
        box = [pad, pad, s - pad, s - pad]
        col = _COLORS[color]
        if shape == "circle":
            d.ellipse(box, fill=col)
        elif shape == "square":
            d.rectangle(box, fill=col)
        elif shape == "triangle":
            d.polygon([(s / 2, pad), (pad, s - pad), (s - pad, s - pad)], fill=col)
        elif shape == "star":
            cx, cy, R, r2 = s / 2, s / 2, (s - 2 * pad) / 2, (s - 2 * pad) / 4
            pts = []
            for k in range(10):
                ang = -np.pi / 2 + k * np.pi / 5
                rad = R if k % 2 == 0 else r2
                pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
            d.polygon(pts, fill=col)
        return img

    @property
    def captions(self):
        """Captions ohne Bild-Rendering (schnell, fuer Vokabular-Aufbau)."""
        return [f"a photo of a {c} {s}" for _, s, c, _ in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        shape_idx, shape, color, jitter = self.items[i]
        img = self._render(shape, color, jitter)
        return {
            "image": _to_tensor(img, self.size),
            "caption": f"a photo of a {color} {shape}",
            "label": shape_idx,
        }


# --------------------------------------------------------------------------- #
#  2) Flickr8k
# --------------------------------------------------------------------------- #
class Flickr8kDataset(Dataset):
    """Erwartet:
        <root>/Images/*.jpg
        <root>/captions.txt   (Format: image,caption  -- eine Zeile je Caption)
    Download-Hinweise: siehe scripts/download_flickr8k.py
    """

    def __init__(self, root: str, image_size: int = 64, split: str = "train",
                 val_frac: float = 0.1, seed: int = 42):
        self.size = image_size
        self.img_dir = os.path.join(root, "Images")
        caps_file = os.path.join(root, "captions.txt")
        if not os.path.isdir(self.img_dir) or not os.path.isfile(caps_file):
            raise FileNotFoundError(
                f"Flickr8k nicht gefunden unter '{root}'. "
                "Siehe scripts/download_flickr8k.py")

        pairs = []
        with open(caps_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.lower().startswith("image,"):
                    continue
                img, _, cap = line.partition(",")
                if cap:
                    pairs.append((img.strip(), cap.strip()))

        # Split auf Bildebene (nicht auf Caption-Ebene), damit kein Leck entsteht.
        images = sorted({p[0] for p in pairs})
        rng = random.Random(seed)
        rng.shuffle(images)
        n_val = int(len(images) * val_frac)
        val_set = set(images[:n_val])
        keep = val_set if split == "val" else (set(images) - val_set)
        self.pairs = [p for p in pairs if p[0] in keep]

    @property
    def captions(self):
        return [c for _, c in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        name, cap = self.pairs[i]
        img = Image.open(os.path.join(self.img_dir, name))
        return {"image": _to_tensor(img, self.size), "caption": cap, "label": -1}


# --------------------------------------------------------------------------- #
#  2b) Flickr8k aus vorbereitetem npy/json-Cache (siehe prep_parquet.py)
# --------------------------------------------------------------------------- #
class CachedFlickr(Dataset):
    """Liest den von prep_parquet.py erzeugten Cache.

    split='train' laedt train0+train1 (6000 Bilder). Pro __getitem__ wird beim
    Training EINE der 5 Captions zufaellig gezogen (Daten-Augmentierung auf der
    Sprachseite). Bei eval=True wird deterministisch caption_0 verwendet, damit
    Retrieval sauber 1:1 messbar ist.
    """

    def __init__(self, root="data/flickr8k_cache", split="train", eval=False, seed=0):
        prefixes = ["train0", "train1"] if split == "train" else [split]
        imgs, caps = [], []
        for p in prefixes:
            imgs.append(np.load(os.path.join(root, f"images_{p}.npy")))
            with open(os.path.join(root, f"caps_{p}.json")) as f:
                caps.extend(json.load(f))
        self.images = np.concatenate(imgs, axis=0)      # [N,64,64,3] uint8
        self.caps = caps                                # Liste[Liste[str]]
        self.eval = eval
        self.rng = random.Random(seed)
        assert len(self.images) == len(self.caps)

    @property
    def captions(self):
        # fuer Vokabular-Aufbau: alle Captions
        return [c for group in self.caps for c in group]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        arr = self.images[i].astype(np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        group = self.caps[i]
        cap = group[0] if self.eval else self.rng.choice(group)
        return {"image": img, "caption": cap, "label": -1}


# --------------------------------------------------------------------------- #
#  Collate: tokenisiert die Captions des Batches
# --------------------------------------------------------------------------- #
class Collate:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        captions = [b["caption"] for b in batch]
        tokens = self.tok.encode_batch(captions)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {"image": images, "text": tokens, "caption": captions, "label": labels}


def all_captions(dataset) -> List[str]:
    """Alle Captions (fuer den Vokabular-Aufbau) -- ohne Bilder zu rendern."""
    if hasattr(dataset, "captions"):
        return list(dataset.captions)
    return [dataset[i]["caption"] for i in range(len(dataset))]
