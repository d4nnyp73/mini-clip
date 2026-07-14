"""Einfacher Wort-Tokenizer.

CLIP nutzt einen BPE-Tokenizer; fuer ein Mini-Projekt reicht ein Whitespace-
Tokenizer mit aus den Trainings-Captions gelerntem Vokabular voellig aus und ist
didaktisch transparenter.
"""
import re
from collections import Counter

import torch

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

_token_re = re.compile(r"[a-z0-9]+")


def basic_tokenize(text: str):
    """Kleinschreibung + nur alphanumerische Tokens."""
    return _token_re.findall(text.lower())


class SimpleTokenizer:
    def __init__(self, max_len: int = 32):
        self.max_len = max_len
        self.stoi = {}
        self.itos = {}

    def build_vocab(self, captions, min_freq: int = 1):
        counter = Counter()
        for c in captions:
            counter.update(basic_tokenize(c))
        vocab = list(SPECIALS) + [w for w, f in counter.most_common() if f >= min_freq]
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    def encode(self, text: str) -> torch.Tensor:
        """Caption -> Tensor fester Laenge [max_len] mit <bos> ... <eos> <pad>..."""
        ids = [self.stoi[BOS]]
        for tok in basic_tokenize(text):
            ids.append(self.stoi.get(tok, self.stoi[UNK]))
            if len(ids) >= self.max_len - 1:
                break
        ids.append(self.stoi[EOS])
        ids += [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids[: self.max_len], dtype=torch.long)

    def encode_batch(self, texts) -> torch.Tensor:
        return torch.stack([self.encode(t) for t in texts])

    # --- Persistenz, damit Eval/Baseline dasselbe Vokabular nutzen ---
    def state_dict(self):
        return {"max_len": self.max_len, "stoi": self.stoi}

    def load_state_dict(self, state):
        self.max_len = state["max_len"]
        self.stoi = state["stoi"]
        self.itos = {i: w for w, i in self.stoi.items()}
        return self
