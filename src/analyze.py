"""Aggregate runs -> results table, training curves, error analysis by length, confident mistakes.
Uses EXACT config names for the two main models (wildcards used to match tuning/ablation runs by accident)."""
import glob, json, os, collections
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from data import load_splits

KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
MAIN = {"CodeBERT": "codebert_L512_head_lr2e-05_bs16", "BiLSTM": "bilstm_L512_head_lr0.001_bs32"}

groups = collections.defaultdict(list)
for f in sorted(glob.glob("results/*/seed*/metrics.json")):
    groups[f.split("/")[1]].append(json.load(open(f)))

rows = ["| config | seeds | " + " | ".join(KEYS) + " | params |", "|" + "---|" * (len(KEYS) + 3)]
for name, rs in groups.items():   # std is the population std (ddof=0) over seeds
    cells = [f"{np.mean([r['test'][k] for r in rs])*100:.1f} ± {np.std([r['test'][k] for r in rs])*100:.1f}" for k in KEYS]
    rows.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + f" | {rs[0]['params']['total']/1e6:.1f}M |")
if os.path.exists("results/length_only/metrics.json"):
    m = json.load(open("results/length_only/metrics.json"))
    rows.append("| length_only | - | " + " | ".join(f"{m[k]*100:.1f}" for k in KEYS) + " | 2 |")
open("results/table.md", "w").write("\n".join(rows)); print("\n".join(rows))

# training curves: the two main configs, every seed (thin lines)
plt.rcParams.update({"font.size": 10})
COL = {"BiLSTM": "tab:orange", "CodeBERT": "tab:blue"}
fig, ax = plt.subplots(1, 3, figsize=(8, 2.6))
for name, cfg in MAIN.items():
    for k, f in enumerate(sorted(glob.glob(f"results/{cfg}/seed*/history.json"))):
        h = json.load(open(f)); x = [e["epoch"] for e in h]
        for a, key in zip(ax, ["train_loss", "val_f1", "val_roc_auc"]):
            a.plot(x, [e[key] for e in h], color=COL[name], alpha=.75, label=name if k == 0 else None)
for a, t in zip(ax, ["Training loss", "Validation F1", "Validation ROC-AUC"]):
    a.set_title(t); a.set_xlabel("epoch"); a.xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].legend(); plt.tight_layout(); plt.savefig("report/fig_curves.pdf"); plt.savefig("report/fig_curves.png", dpi=200)

# error analysis by function length (all seeds), confident mistakes (seed 0)
test = load_splits()["test"]
bins = [0, 128, 256, 512, 1024, 2048, 10**9]; lab = ["<128", "128-256", "256-512", "512-1k", "1k-2k", ">2k"]
for name, cfg in MAIN.items():
    fs = sorted(glob.glob(f"results/{cfg}/seed*/test_preds.npz"))
    if not fs: continue
    D = [np.load(f) for f in fs]
    assert all((d["idx"] == D[0]["idx"]).all() for d in D)
    y, fl = D[0]["y"], D[0]["full_len"]
    ok = np.stack([((d["prob"] >= .5).astype(int) == d["y"]) for d in D])
    print(f"\n{name} ({cfg}), {len(D)} seeds: accuracy by full function length (%)")
    for lo, hi, l in zip(bins[:-1], bins[1:], lab):
        s = (fl >= lo) & (fl < hi)
        if s.sum(): print(f"  {l:8s} seed0 {ok[0][s].mean()*100:5.1f}   mean over seeds {ok[:, s].mean()*100:5.1f}   (n={s.sum()})")
    fnr = [(((d['prob'] < .5) & (d['y'] == 1)).sum() / (d['y'] == 1).sum()) for d in D]
    print("  false-negative rate per seed (%):", [round(x * 100, 1) for x in fnr])
    d = D[0]; pred = (d["prob"] >= .5).astype(int); okk = pred == d["y"]
    conf = np.abs(d["prob"] - .5); wrong = np.where(~okk)[0]; top = wrong[np.argsort(-conf[wrong])][:20]
    with open(f"results/{name.lower()}_confident_errors.txt", "w") as fh:
        for i in top:
            j = d["idx"][i]
            fh.write(f"=== idx {j} label={d['y'][i]} p(vuln)={d['prob'][i]:.3f} len={d['full_len'][i]}\n{test['func'][j]}\n\n")
