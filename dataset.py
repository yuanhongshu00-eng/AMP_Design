import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import clean_sequence, encode_sequence


def load_sequences(path, sequence_column=None, max_len=50):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        frame = pd.read_csv(path)
        if sequence_column is None:
            values = frame.iloc[:, 0].tolist()
        else:
            values = frame[sequence_column].tolist()
    else:
        with open(path, "r", encoding="utf-8") as handle:
            values = [line.strip() for line in handle]

    sequences = []
    for value in values:
        seq = clean_sequence(value)
        if seq is not None and len(seq) <= max_len:
            sequences.append(seq)
    return sequences


class CachedESM2Dataset(Dataset):
    def __init__(self, cache):
        self.embeddings = cache["embeddings"].float()
        self.attention_mask = cache["attention_mask"].long()
        self.token_ids = cache["token_ids"].long()
        self.lengths = cache["lengths"].long()

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "attention_mask": self.attention_mask[idx],
            "token_ids": self.token_ids[idx],
            "lengths": self.lengths[idx],
        }


class CompressedLatentDataset(Dataset):
    def __init__(self, cache):
        self.z = cache["z"].float()
        self.attention_mask = cache["attention_mask"].long()

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return {"z": self.z[idx], "attention_mask": self.attention_mask[idx]}


def build_internal_tokens(sequences, max_len):
    token_ids = torch.stack([encode_sequence(seq, max_len) for seq in sequences])
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    attention_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return token_ids, lengths, attention_mask.long()
