import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import CachedESM2Dataset
from .decoder import ESM2LatentDecoder
from .normalizer import ESM2LatentNormalizer
from .utils import PAD_ID, VOCAB, decode_token_ids, ensure_parent_dir, sequence_accuracy, token_accuracy, valid_reconstruction_ratio


def _split(dataset, val_ratio, seed):
    val_size = max(1, int(len(dataset) * val_ratio)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size == 0:
        return dataset, dataset
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def build_decoder_from_config(config, vocab_size):
    return ESM2LatentDecoder(
        input_dim=config["input_dim"],
        vocab_size=vocab_size,
        decoder_hidden_size=config.get("decoder_hidden_size", 512),
        decoder_layers=config.get("decoder_layers", 4),
        decoder_heads=config.get("decoder_heads", 8),
        decoder_dropout=config.get("decoder_dropout", 0.1),
    )


def load_decoder_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_decoder_from_config(ckpt["config"], len(ckpt["vocab"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device), ckpt


def _run_epoch(model, loader, normalizer, criterion, device, opt=None):
    is_train = opt is not None
    model.train(is_train)
    total_loss = 0.0
    total_tok = 0.0
    total_seq = 0.0
    total_valid = 0.0
    batches = 0
    for batch in tqdm(loader, leave=False):
        h = batch["embeddings"].to(device)
        mask = batch["attention_mask"].to(device)
        tokens = batch["token_ids"].to(device)
        h_norm = normalizer.transform(h)
        if is_train:
            opt.zero_grad()
        logits = model(h_norm, attention_mask=mask)
        loss = criterion(logits.view(-1, logits.shape[-1]), tokens.view(-1))
        if is_train:
            loss.backward()
            opt.step()
        total_loss += loss.item()
        total_tok += token_accuracy(logits, tokens, mask)
        total_seq += sequence_accuracy(logits, tokens, mask)
        total_valid += valid_reconstruction_ratio(logits, mask)
        batches += 1
    denom = max(1, batches)
    return total_loss / denom, total_tok / denom, total_seq / denom, total_valid / denom


def train_esm2_decoder(args):
    cache = torch.load(args.esm2_cache_path, map_location="cpu")
    dataset = CachedESM2Dataset(cache)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if os.path.exists(args.esm2_normalizer_path):
        normalizer = ESM2LatentNormalizer.load(args.esm2_normalizer_path)
    else:
        normalizer = ESM2LatentNormalizer(
            clip_value=args.flow_clip_value,
            use_tanh_smoothing=args.use_tanh_smoothing,
        )
        normalizer.fit(cache["embeddings"], cache["attention_mask"])
        ensure_parent_dir(args.esm2_normalizer_path)
        normalizer.save(args.esm2_normalizer_path)
        print(f"Saved normalizer to {args.esm2_normalizer_path}")

    config = {
        "input_dim": int(cache["hidden_size"]),
        "decoder_hidden_size": args.decoder_hidden_size,
        "decoder_layers": args.decoder_layers,
        "decoder_heads": args.decoder_heads,
        "decoder_dropout": args.decoder_dropout,
    }
    model = build_decoder_from_config(config, len(VOCAB)).to(device)
    train_set, val_set = _split(dataset, args.val_ratio, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.decoder_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.decoder_batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    opt = AdamW(model.parameters(), lr=args.decoder_lr, weight_decay=args.decoder_weight_decay)

    best = float("inf")
    ensure_parent_dir(args.esm2_decoder_path)
    for epoch in range(args.decoder_epoch):
        train = _run_epoch(model, train_loader, normalizer, criterion, device, opt)
        with torch.no_grad():
            val = _run_epoch(model, val_loader, normalizer, criterion, device)
        print(
            f"Epoch {epoch}: train_loss={train[0]:.5f} val_loss={val[0]:.5f} "
            f"token_acc={val[1]:.4f} seq_acc={val[2]:.4f} valid_ratio={val[3]:.4f}"
        )
        print("Rule: if decoder(original h_norm) token accuracy is poor, do not continue to compressor.")
        if val[0] < best:
            best = val[0]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "vocab": VOCAB,
                    "best_val_loss": best,
                },
                args.esm2_decoder_path,
            )
    print(f"Saved best decoder to {args.esm2_decoder_path}")


def reconstruct_from_cached_embeddings(cache_path, normalizer_path, decoder_path, device="cuda", limit=8):
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    cache = torch.load(cache_path, map_location="cpu")
    normalizer = ESM2LatentNormalizer.load(normalizer_path)
    decoder, _ = load_decoder_checkpoint(decoder_path, device)
    decoder.eval()
    h = cache["embeddings"][:limit].to(device)
    mask = cache["attention_mask"][:limit].to(device)
    lengths = cache["lengths"][:limit]
    with torch.no_grad():
        logits = decoder(normalizer.transform(h), attention_mask=mask)
    return [decode_token_ids(row, length) for row, length in zip(logits.argmax(-1), lengths)]
