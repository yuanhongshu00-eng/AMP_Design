import os
import random

import numpy as np
import torch


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
VOCAB_TOKENS = SPECIAL_TOKENS + list(AMINO_ACIDS)
VOCAB = {token: idx for idx, token in enumerate(VOCAB_TOKENS)}
ID_TO_TOKEN = {idx: token for token, idx in VOCAB.items()}
PAD_ID = VOCAB["<pad>"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def clean_sequence(seq):
    seq = str(seq).strip().upper()
    if not seq:
        return None
    if any(ch not in AMINO_ACIDS for ch in seq):
        return None
    return seq


def encode_sequence(seq, max_len):
    ids = torch.full((max_len,), PAD_ID, dtype=torch.long)
    for i, aa in enumerate(seq[:max_len]):
        ids[i] = VOCAB[aa]
    return ids


def decode_token_ids(token_ids, length=None):
    ids = token_ids.detach().cpu().tolist()
    if length is not None:
        ids = ids[: int(length)]
    aas = []
    for idx in ids:
        token = ID_TO_TOKEN.get(int(idx), "")
        if token in AMINO_ACIDS:
            aas.append(token)
    return "".join(aas)


def masked_mean(values, attention_mask, eps=1e-8):
    mask = attention_mask.to(values.device).float()
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def masked_mse(pred, target, attention_mask):
    return masked_mean((pred - target) ** 2, attention_mask)


def token_accuracy(logits, token_ids, attention_mask, pad_id=PAD_ID):
    pred = logits.argmax(dim=-1)
    valid = (attention_mask == 1) & (token_ids != pad_id)
    denom = valid.sum().clamp_min(1)
    return ((pred == token_ids) & valid).sum().float().div(denom).item()


def sequence_accuracy(logits, token_ids, attention_mask, pad_id=PAD_ID):
    pred = logits.argmax(dim=-1)
    valid = (attention_mask == 1) & (token_ids != pad_id)
    per_seq = ((pred == token_ids) | ~valid).all(dim=1)
    return per_seq.float().mean().item()


def valid_reconstruction_ratio(logits, attention_mask):
    pred = logits.argmax(dim=-1).detach().cpu()
    mask = attention_mask.detach().cpu()
    valid_ids = {VOCAB[aa] for aa in AMINO_ACIDS}
    ratios = []
    for row, row_mask in zip(pred, mask):
        ids = row[row_mask.bool()].tolist()
        if not ids:
            ratios.append(0.0)
        else:
            ratios.append(sum(int(i) in valid_ids for i in ids) / len(ids))
    return float(np.mean(ratios)) if ratios else 0.0
