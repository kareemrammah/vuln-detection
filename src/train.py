"""Train one model/seed/config; select checkpoint by best validation F1; evaluate once on test.
Example:
  python src/train.py --model codebert --max_len 512 --strategy head --seed 0
  python src/train.py --model bilstm --seed 0
"""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from data import load_splits, get_tokenizer, tokenize_all, make_loader
from models import BiLSTMClassifier, CodeBERTClassifier, codebert_word_embeddings, count_params


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p)}


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, Y, FL, IDX = [], [], [], []
    for b in loader:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logit = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
        P.append(torch.sigmoid(logit.float()).cpu()); Y.append(b["labels"]); FL.append(b["full_len"]); IDX.append(b["idx"])
    return (torch.cat(P).numpy(), torch.cat(Y).numpy().astype(int), torch.cat(FL).numpy(), torch.cat(IDX).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bilstm", "bilstm_pt", "codebert"], required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--strategy", choices=["head", "headtail"], default="head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_top_k", type=int, default=None, help="CodeBERT: only train top k layers")
    ap.add_argument("--no_dedupe", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cb = a.model == "codebert"
    lr = a.lr or (2e-5 if cb else 1e-3)
    bs = a.bs or (16 if cb else 32)
    epochs = a.epochs or (5 if cb else 10)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = f"{a.model}_L{a.max_len}_{a.strategy}" + (f"_top{a.train_top_k}" if a.train_top_k else "") + f"_lr{lr:g}_bs{bs}"
    out = os.path.join(a.out, name, f"seed{a.seed}"); os.makedirs(out, exist_ok=True)

    splits = load_splits(dedupe=not a.no_dedupe)
    tok = get_tokenizer()
    ids = tokenize_all(splits, tok) if not a.no_dedupe else None
    if ids is None:  # cache is built from the deduplicated train split; recompute if not deduping
        ids = {sp: tok(d["func"], add_special_tokens=False, truncation=False)["input_ids"] for sp, d in splits.items()}
    mk = lambda sp, sh: make_loader(ids[sp], splits[sp]["y"], tok, a.max_len, a.strategy, bs, sh)
    tr, va, te = mk("train", True), mk("validation", False), mk("test", False)

    if cb:
        model = CodeBERTClassifier(train_top_k=a.train_top_k)
    elif a.model == "bilstm_pt":
        model = BiLSTMClassifier(pretrained_emb=codebert_word_embeddings())
    else:
        model = BiLSTMClassifier()
    model.to(dev)
    params = count_params(model); print(params)

    opt_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=lr, weight_decay=0.01) if cb else torch.optim.Adam(opt_params, lr=lr)
    total = epochs * len(tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total) if cb else None
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()

    best, hist = -1, []
    for ep in range(epochs):
        model.train(); t0, run = time.time(), []
        for b in tqdm(tr, desc=f"epoch {ep+1}/{epochs}"):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            loss = lossf(logit.float(), b["labels"].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(opt_params, 1.0)
            scaler.step(opt); scaler.update()
            if sched: sched.step()
            run.append(loss.item())
        p, y, _, _ = predict(model, va, dev)
        vm = metrics(y, p)
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(run)), **{f"val_{k}": v for k, v in vm.items()},
                     "sec": time.time() - t0})
        print(hist[-1])
        if vm["f1"] > best:
            best = vm["f1"]; torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=1)

    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev))
    p, y, fl, idx = predict(model, te, dev)
    res = {"config": vars(a) | {"lr": lr, "bs": bs, "epochs": epochs, "name": name},
           "test": metrics(y, p), "best_val_f1": best, "params": params}
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    np.savez(os.path.join(out, "test_preds.npz"), prob=p, y=y, full_len=fl, idx=idx)
    print("TEST", res["test"])
    os.remove(os.path.join(out, "best.pt")) if os.environ.get("DROP_CKPT") else None


if __name__ == "__main__":
    main()
