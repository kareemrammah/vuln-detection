"""BiLSTM baseline (random or CodeBERT-initialised embeddings) and fine-tuned CodeBERT classifier."""
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

CODEBERT = "microsoft/codebert-base"


class BiLSTMClassifier(nn.Module):
    """embeddings(256) -> 2-layer BiLSTM(256/dir) -> masked max-pool over time (512) -> dropout -> linear."""

    def __init__(self, vocab_size=50265, pad_id=1, emb_dim=256, hidden=256, layers=2,
                 dropout=0.3, pretrained_emb=None):
        super().__init__()
        if pretrained_emb is None:                       # random init (baseline)
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
            self.proj = None
        else:                                            # BiLSTM-PT: CodeBERT embeddings (768) + learned projection
            self.emb = nn.Embedding.from_pretrained(pretrained_emb.clone(), freeze=False, padding_idx=pad_id)
            self.proj = nn.Linear(pretrained_emb.size(1), emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        if self.proj is not None:
            x = self.proj(x)
        lengths = attention_mask.sum(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=input_ids.size(1))
        h = h.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))   # ignore padding
        h = h.max(dim=1).values
        return self.out(self.drop(h)).squeeze(-1)        # logit; sigmoid applied inside the loss / at eval


def codebert_word_embeddings():
    return AutoModel.from_pretrained(CODEBERT).embeddings.word_embeddings.weight.detach()


class CodeBERTClassifier(nn.Module):
    """CodeBERT encoder -> final hidden state of [CLS] (768) -> dropout 0.1 -> linear."""

    def __init__(self, train_top_k=None, dropout=0.1, encoder=None):
        super().__init__()
        self.enc = encoder if encoder is not None else AutoModel.from_pretrained(CODEBERT)
        d = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 1)
        if train_top_k is not None:                      # layer-freezing ablation (e.g. top 4 of 12)
            for p in self.enc.embeddings.parameters():
                p.requires_grad = False
            n = len(self.enc.encoder.layer)
            for i, layer in enumerate(self.enc.encoder.layer):
                for p in layer.parameters():
                    p.requires_grad = i >= n - train_top_k

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.out(self.drop(h)).squeeze(-1)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parts = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()}
    return {"total": total, "trainable": train, "by_module": parts}
