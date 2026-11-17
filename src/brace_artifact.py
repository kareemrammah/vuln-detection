"""Formatting-artifact check: how often a function ends without a closing brace, by class,
and how each model's accuracy differs on test functions with / without that artifact.
Run:  python src/brace_artifact.py"""
import glob, numpy as np
from data import load_splits

S = load_splits(dedupe=False)
no_final = lambda f: not f.rstrip().endswith("}")
unbalanced = lambda f: f.count("{") != f.count("}")
for sp in ("train", "test"):
    y = np.array(S[sp]["y"])
    for name, fn in (("no_final_brace", no_final), ("unbalanced_braces", unbalanced)):
        v = np.array([fn(f) for f in S[sp]["func"]])
        print(f"{sp:5s} {name:17s} vulnerable {v[y==1].mean()*100:5.2f}%  non-vulnerable {v[y==0].mean()*100:5.2f}%  (flagged: {v.sum()})")

flag = np.array([no_final(f) for f in S["test"]["func"]])
print(f"\ntest functions without a final brace: {flag.sum()} of {len(flag)}  "
      f"(labels: {np.array(S['test']['y'])[flag].mean()*100:.0f}% vulnerable)")
CFG = {"BiLSTM": "bilstm_L512_head_lr0.001_bs32",
       "CodeBERT lr2e-5": "codebert_L512_head_lr2e-05_bs16",
       "CodeBERT lr3e-5": "codebert_L512_head_lr3e-05_bs16"}
for name, cfg in CFG.items():
    rows = []
    for f in sorted(glob.glob(f"results/{cfg}/seed*/test_preds.npz")):
        d = np.load(f)
        ok = np.zeros(len(flag), bool); ok[d["idx"]] = ((d["prob"] >= .5).astype(int) == d["y"])
        rows.append((ok.mean() * 100, ok[flag].mean() * 100, ok[~flag].mean() * 100))
    if rows:
        a = np.mean(rows, axis=0)
        print(f"{name:16s} seeds={len(rows)}  accuracy all {a[0]:.1f}%  | no final brace {a[1]:.1f}%  | with final brace {a[2]:.1f}%")
