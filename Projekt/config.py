"""Zentrale Konfiguration fuer Mini-CLIP.

Alle Hyperparameter an einer Stelle, damit Experimente reproduzierbar sind und
in der Dokumentation eindeutig referenziert werden koennen.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Daten ---
    image_size: int = 64            # Kantenlaenge (quadratisch). Klein halten fuer CPU.
    max_text_len: int = 32          # max. Tokens pro Caption (inkl. <bos>/<eos>)
    min_word_freq: int = 1          # Mindesthaeufigkeit, damit ein Wort ins Vokabular kommt

    # --- Modell (bewusst klein -> CPU-tauglich) ---
    embed_dim: int = 256            # gemeinsame Embedding-Dimension (shared space)
    # Text-Encoder
    text_width: int = 128           # d_model des Text-Transformers
    text_heads: int = 4
    text_layers: int = 2
    # Bild-Encoder
    vision_channels: tuple = (32, 64, 128, 256)  # Kanaele je Conv-Block

    # --- Kontrastiver Loss ---
    init_logit_scale: float = 0.07  # Start-Temperatur (wie im CLIP-Paper)
    max_logit_scale: float = 100.0  # logit_scale wird auf <= 100 geclippt

    # --- Training ---
    batch_size: int = 128
    epochs: int = 15
    lr: float = 5e-4
    weight_decay: float = 0.2
    warmup_steps: int = 50
    seed: int = 42
    num_workers: int = 0            # 0 = sicher auf CPU/macOS

    # --- Pfade ---
    data_root: str = "data"
    ckpt_path: str = "checkpoints/mini_clip.pt"
    device: str = "cpu"
