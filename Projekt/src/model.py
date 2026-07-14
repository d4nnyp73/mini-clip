"""Mini-CLIP: Bild-Encoder + Text-Encoder + gemeinsamer Embedding-Raum.

Architektur folgt dem CLIP-Prinzip (Radford et al., 2021), nur stark verkleinert:
  - Bild-Encoder:  kleines CNN  (statt ResNet-50 / ViT-B)
  - Text-Encoder:  kleiner Transformer (statt 63M-Param-Transformer)
  - beide werden per linearer Projektion in denselben d-dim Raum abgebildet,
    L2-normalisiert und ueber eine gelernte Temperatur (logit_scale) verglichen.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Bild-Encoder
# --------------------------------------------------------------------------- #
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, in_c = [], 3
        for out_c in channels:
            layers += [
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_c, embed_dim)

    def forward(self, x):                       # x: [B,3,H,W]
        h = self.features(x)
        h = self.pool(h).flatten(1)             # [B, C]
        return self.proj(h)                     # [B, embed_dim]


# --------------------------------------------------------------------------- #
#  Text-Encoder
# --------------------------------------------------------------------------- #
class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embed_dim: int,
                 width: int = 128, heads: int = 4, layers: int = 2,
                 max_len: int = 32):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, width))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.ln = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim)
        nn.init.normal_(self.pos_emb, std=0.01)

    def forward(self, tokens):                  # tokens: [B, L]
        pad_mask = tokens == self.pad_id        # True an Padding-Positionen
        h = self.token_emb(tokens) + self.pos_emb[:, : tokens.size(1)]
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        # Mean-Pooling ueber Nicht-Padding-Tokens
        mask = (~pad_mask).unsqueeze(-1).float()
        h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(h)                     # [B, embed_dim]


# --------------------------------------------------------------------------- #
#  Mini-CLIP
# --------------------------------------------------------------------------- #
class MiniCLIP(nn.Module):
    def __init__(self, cfg, vocab_size: int, pad_id: int):
        super().__init__()
        self.image_encoder = ImageEncoder(cfg.embed_dim, cfg.vision_channels)
        self.text_encoder = TextEncoder(
            vocab_size, pad_id, cfg.embed_dim, cfg.text_width,
            cfg.text_heads, cfg.text_layers, cfg.max_text_len)
        # Temperatur als log-Parameter (wie im CLIP-Code), Start: 1/0.07
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / cfg.init_logit_scale)))
        self.max_logit_scale = math.log(cfg.max_logit_scale)

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.text_encoder(tokens), dim=-1)

    def forward(self, images, tokens):
        img = self.encode_image(images)
        txt = self.encode_text(tokens)
        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()
        logits_per_image = scale * img @ txt.t()       # [B, B]
        return logits_per_image, logits_per_image.t()


def clip_loss(logits_per_image, logits_per_text):
    """Symmetrischer InfoNCE-Loss (Kern von CLIP).

    Auf der Diagonalen stehen die zusammengehoerigen (Bild, Text)-Paare des
    Batches; alle anderen Eintraege sind In-Batch-Negative. Wir maximieren die
    Aehnlichkeit der Paare in beide Richtungen (Bild->Text und Text->Bild).
    """
    n = logits_per_image.size(0)
    targets = torch.arange(n, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2
