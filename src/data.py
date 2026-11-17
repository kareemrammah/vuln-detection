"""Data loading, deduplication, tokenization and batching for Devign (CodeXGLUE split)."""
import os, pickle
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

TOK_NAME = "microsoft/codebert-base"
CACHE = "cache"


def normalize(s):
    return " ".join(s.split())


def load_splits(dedupe=True):
    """Returns {'train'|'validation'|'test': {'func': [...], 'y': [...]}}.
    If dedupe: drop duplicate training functions and any training function that also
    appears in validation/test (whitespace-normalised exact match), as in the proposal."""
    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    out = {sp: {"func": list(ds[sp]["func"]), "y": [int(t) for t in ds[sp]["target"]]}
           for sp in ("train", "validation", "test")}
    if dedupe:
        held = {normalize(f) for sp in ("validation", "test") for f in out[sp]["func"]}
        seen, keep = set(), []
        for i, f in enumerate(out["train"]["func"]):
            k = normalize(f)
            if k in held or k in seen:
                continue
            seen.add(k)
            keep.append(i)
        out["train"] = {"func": [out["train"]["func"][i] for i in keep],
                        "y": [out["train"]["y"][i] for i in keep]}
    return out


def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOK_NAME)


def tokenize_all(splits, tok):
    """Full (untruncated) BPE ids without special tokens, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "tokens.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    ids = {}
    for sp, d in splits.items():
        ids[sp] = tok(d["func"], add_special_tokens=False, truncation=False)["input_ids"]
    pickle.dump(ids, open(path, "wb"))
    return ids


def truncate(ids, L, strategy, cls_id, sep_id):
    """L counts special tokens, so content budget is L-2 (510 for L=512)."""
    budget = L - 2
    if len(ids) > budget:
        if strategy == "head":
            ids = ids[:budget]
        elif strategy == "headtail":
            h = budget // 2
            ids = ids[:h] + ids[len(ids) - (budget - h):]
        else:
            raise ValueError(strategy)
    return [cls_id] + ids + [sep_id]


class FuncDataset(Dataset):
    def __init__(self, ids, y):
        self.ids, self.y = ids, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ids[i], self.y[i], len(self.ids[i]), i


def make_loader(ids, y, tok, L, strategy, bs, shuffle):
    cls_id, sep_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    def collate(batch):
        seqs = [truncate(b[0], L, strategy, cls_id, sep_id) for b in batch]
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        m = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            m[i, :len(s)] = 1
        return {"input_ids": x, "attention_mask": m,
                "labels": torch.tensor([b[1] for b in batch], dtype=torch.float),
                "full_len": torch.tensor([b[2] for b in batch]),
                "idx": torch.tensor([b[3] for b in batch])}

    return DataLoader(FuncDataset(ids, y), batch_size=bs, shuffle=shuffle, collate_fn=collate)
