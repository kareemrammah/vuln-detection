"""EDA for the Data section: split sizes, class balance, length statistics, duplicates, Figure 1."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_splits, get_tokenizer, tokenize_all, normalize

raw = load_splits(dedupe=False)
print({k: (len(v["y"]), round(float(np.mean(v["y"])), 3)) for k, v in raw.items()})
tr = {normalize(f) for f in raw["train"]["func"]}
print("unique train:", len(tr), "test dup:", sum(normalize(f) in tr for f in raw["test"]["func"]),
      "val dup:", sum(normalize(f) in tr for f in raw["validation"]["func"]))
clean = load_splits(dedupe=True); print("train after dedupe:", len(clean["train"]["y"]))
ids = tokenize_all(clean, get_tokenizer())
L = np.array([len(x) for sp in ids for x in ids[sp]])
print("median", np.median(L), "mean", L.mean(), "p90", np.percentile(L, 90), "max", L.max(), "frac>512", (L > 512).mean())
for c in (0, 1):
    print("class", c, "train median len", np.median([len(x) for x, y in zip(ids["train"], clean["train"]["y"]) if y == c]))
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(8, 2.7))
ax[0].bar(list(raw), [np.mean(v["y"]) for v in raw.values()]); ax[0].axhline(.5, ls="--", c="gray")
ax[0].set_ylim(0, 1); ax[0].set_ylabel("Fraction vulnerable"); ax[0].set_title("Class balance per split")
ax[1].hist(np.clip(L, 0, 2000), bins=80); ax[1].axvline(512, ls="--", c="k", label="512-token limit")
ax[1].set_xlabel("Function length (tokens, clipped at 2000)", fontsize=9); ax[1].set_ylabel("Number of functions"); ax[1].legend()
plt.tight_layout(); plt.savefig("report/eda.png", dpi=200)
