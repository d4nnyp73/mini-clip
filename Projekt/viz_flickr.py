"""Erzeugt Ergebnis-Abbildungen fuer den Flickr8k-Lauf:
  assets/flickr_curve.png   -- Train-Loss vs. Test-Recall@10 ueber 30 Epochen
  assets/flickr_compare.png -- Recall-Vergleich Mini-CLIP vs. pretrained CLIP vs. Zufall

Die Serien stammen aus den protokollierten Trainingslaeufen (train_flickr.py)
bzw. baseline_flickr_onnx.py und sind hier zur Reproduktion der Plots hinterlegt.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

# --- Trainingsverlauf (Test-Split, 1000 Bilder) ---
epochs = list(range(1, 31))
loss = [5.577,5.354,5.188,5.025,4.824,4.515,4.193,3.880,3.648,3.416,3.117,2.856,
        2.628,2.487,2.337,2.148,1.990,1.856,1.782,1.715,1.594,1.501,1.428,1.395,
        1.372,1.300,1.254,1.222,1.227,1.242]
i2t_r10 = [1.2,3.8,4.7,4.4,5.7,5.6,5.5,5.0,5.7,6.1,5.0,5.4,4.0,4.5,6.4,4.9,5.8,5.2,
           5.0,5.9,5.0,5.5,5.5,5.3,6.2,5.2,5.8,6.1,6.0,6.0]

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(epochs, loss, "o-", color="tab:red", label="Train-Loss")
ax1.set_xlabel("Epoche"); ax1.set_ylabel("Train-Loss", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax2 = ax1.twinx()
ax2.plot(epochs, i2t_r10, "s-", color="tab:blue", label="Test i2t Recall@10")
ax2.axhline(1.0, ls="--", color="gray", lw=1, label="Zufall R@10 = 1%")
ax2.set_ylabel("Test Recall@10 (%)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue"); ax2.set_ylim(0, 12)
ax1.set_title("Mini-CLIP auf Flickr8k: Loss fällt, Test-Recall stagniert (Overfitting)")
fig.tight_layout(); fig.savefig("assets/flickr_curve.png", dpi=130); plt.close(fig)
print("assets/flickr_curve.png")

# --- Vergleich auf 1000 Test-Bildern ---
ks = ["R@1", "R@5", "R@10"]
chance = [0.1, 0.5, 1.0]
mini = [0.9, 2.9, 6.0]            # bestes i2t unseres Mini-CLIP
pre = [40.0, 66.5, 76.7]          # pretrained CLIP ViT-B/32 (ONNX int8)
x = range(len(ks)); w = 0.27
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar([i - w for i in x], chance, w, label="Zufall", color="lightgray")
ax.bar(list(x), mini, w, label="Mini-CLIP (6k Paare, from scratch)", color="tab:blue")
ax.bar([i + w for i in x], pre, w, label="Pretrained CLIP (400M Paare)", color="tab:green")
for i in x:
    ax.text(i, mini[i] + 1, f"{mini[i]}", ha="center", fontsize=8)
    ax.text(i + w, pre[i] + 1, f"{pre[i]}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(ks)
ax.set_ylabel("Image->Text Recall (%)"); ax.set_ylim(0, 90)
ax.set_title("Flickr8k Retrieval (1000 Test-Bilder): der Effekt der Skalierung")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("assets/flickr_compare.png", dpi=130); plt.close(fig)
print("assets/flickr_compare.png")
