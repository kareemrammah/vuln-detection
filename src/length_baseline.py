"""Trivial reference: logistic regression on log(token length) only."""
import json, numpy as np, os
from sklearn.linear_model import LogisticRegression
from data import load_splits, get_tokenizer, tokenize_all
from train import metrics

splits = load_splits(); ids = tokenize_all(splits, get_tokenizer())
f = lambda sp: np.log1p([len(x) for x in ids[sp]]).reshape(-1, 1)
clf = LogisticRegression().fit(f("train"), splits["train"]["y"])
m = metrics(np.array(splits["test"]["y"]), clf.predict_proba(f("test"))[:, 1])
os.makedirs("results/length_only", exist_ok=True)
json.dump(m, open("results/length_only/metrics.json", "w"), indent=1); print(m)
