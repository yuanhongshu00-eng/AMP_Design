import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .compressor import ESM2Compressor, ESM2Decompressor
from .dataset import CachedESM2Dataset
from .normalizer import ESM2LatentNormalizer
from .train_decoder import load_decoder_checkpoint
from .utils import PAD_ID, VOCAB, decode_token_ids, ensure_parent_dir, masked_mse, sequence_accuracy, token_accuracy


def build_compressor_pair_from_config(config):
    compressor = ESM2Compressor(
        input_dim=config["input_dim"],
        compressed_dim=config.get("compressed_dim", 128),
        compressor_hidden_size=config.get("compressor_hidden_size", 512),
        compressor_pre_layers=config.get("compressor_pre_layers", 2),
        compressor_post_layers=config.get("compressor_post_layers", 2),
        compressor_heads=config.get("compressor_heads", 8),
        compressor_dropout=config.get("compressor_dropout", 0.1),
    )
    decompressor = ESM2Decompressor(
        output_dim=config["input_dim"],
        compressed_dim=config.get("compressed_dim", 128),
        compressor_hidden_size=config.get("compressor_hidden_size", 512),
        decompressor_pre_layers=config.get("decompressor_pre_layers", 2),
        decompressor_post_layers=config.get("decompressor_post_layers", 2),
        compressor_heads=config.get("compressor_heads", 8),
        compressor_dropout=config.get("compressor_dropout", 0.1),
    )
    return compressor, decompressor


def load_compressor_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    compressor, decompressor = build_compressor_pair_from_config(ckpt["config"])
    compressor.load_state_dict(ckpt["compressor_state_dict"])
    decompressor.load_state_dict(ckpt["decompressor_state_dict"])
    return compressor.to(device), decompressor.to(device), ckpt


def _masked_cosine_loss(h_rec, h_norm, attention_mask):
    cos = F.cosine_similarity(h_rec, h_norm, dim=-1)
    mask = attention_mask.float().to(cos.device)
    return ((1.0 - cos) * mask).sum() / mask.sum().clamp_min(1.0)


def _run_epoch(compressor, decompressor, decoder, loader, normalizer, criterion, args, device, epoch, opt=None):
    is_train = opt is not None
    compressor.train(is_train)
    decompressor.train(is_train)
    total = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "ce": 0.0, "tok": 0.0, "seq": 0.0}
    batches = 0
    for batch in tqdm(loader, leave=False):
        h = batch["embeddings"].to(device)
        mask = batch["attention_mask"].to(device)
        tokens = batch["token_ids"].to(device)
        h_norm = normalizer.transform(h)
        if is_train:
            opt.zero_grad()
        z = compressor(h_norm, attention_mask=mask)
        h_rec = decompressor(z, attention_mask=mask)
        logits = decoder(h_rec, attention_mask=mask)
        mse = masked_mse(h_rec, h_norm, mask)
        cos = _masked_cosine_loss(h_rec, h_norm, mask)
        ce = criterion(logits.view(-1, logits.shape[-1]), tokens.view(-1))
        loss = args.mse_weight * mse + args.cosine_weight * cos
        if epoch >= args.ce_warmup_epoch:
            loss = loss + args.ce_weight * ce
        if is_train:
            loss.backward()
            opt.step()
        total["loss"] += loss.item()
        total["mse"] += mse.item()
        total["cos"] += (1.0 - cos.item())
        total["ce"] += ce.item()
        total["tok"] += token_accuracy(logits, tokens, mask)
        total["seq"] += sequence_accuracy(logits, tokens, mask)
        batches += 1
    return {key: value / max(1, batches) for key, value in total.items()}


def train_esm2_compressor(args):
    cache = torch.load(args.esm2_cache_path, map_location="cpu")
    dataset = CachedESM2Dataset(cache)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    normalizer = ESM2LatentNormalizer.load(args.esm2_normalizer_path)
    decoder, _ = load_decoder_checkpoint(args.esm2_decoder_path, device)
    decoder.eval()
    decoder.requires_grad_(False)

    config = {
        "input_dim": int(cache["hidden_size"]),
        "compressed_dim": args.compressed_dim,
        "compressor_hidden_size": args.compressor_hidden_size,
        "compressor_pre_layers": args.compressor_pre_layers,
        "compressor_post_layers": args.compressor_post_layers,
        "decompressor_pre_layers": args.decompressor_pre_layers,
        "decompressor_post_layers": args.decompressor_post_layers,
        "compressor_heads": args.compressor_heads,
        "compressor_dropout": args.compressor_dropout,
    }
    compressor, decompressor = build_compressor_pair_from_config(config)
    compressor, decompressor = compressor.to(device), decompressor.to(device)
    val_size = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_set, val_set = (random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)) if val_size else (dataset, dataset))
    train_loader = DataLoader(train_set, batch_size=args.compressor_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.compressor_batch_size, shuffle=False)
    opt = AdamW(list(compressor.parameters()) + list(decompressor.parameters()), lr=args.compressor_lr, weight_decay=args.compressor_weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    best = float("inf")
    ensure_parent_dir(args.esm2_compressor_path)
    for epoch in range(args.compressor_epoch):
        train = _run_epoch(compressor, decompressor, decoder, train_loader, normalizer, criterion, args, device, epoch, opt)
        with torch.no_grad():
            val = _run_epoch(compressor, decompressor, decoder, val_loader, normalizer, criterion, args, device, epoch)
        print(
            f"Epoch {epoch}: train_loss={train['loss']:.5f} val_loss={val['loss']:.5f} "
            f"val_mse={val['mse']:.5f} val_cos_sim={val['cos']:.4f} val_ce={val['ce']:.5f} "
            f"token_acc={val['tok']:.4f} seq_acc={val['seq']:.4f}"
        )
        print("Rule: if compression reconstruction token accuracy is poor, do not continue to flow.")
        if val["loss"] < best:
            best = val["loss"]
            torch.save(
                {
                    "compressor_state_dict": compressor.state_dict(),
                    "decompressor_state_dict": decompressor.state_dict(),
                    "config": config,
                    "best_val_loss": best,
                },
                args.esm2_compressor_path,
            )
    print(f"Saved best compressor/decompressor to {args.esm2_compressor_path}")


def reconstruct_after_compression(cache_path, normalizer_path, decoder_path, compressor_path, device="cuda", limit=8):
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    cache = torch.load(cache_path, map_location="cpu")
    normalizer = ESM2LatentNormalizer.load(normalizer_path)
    decoder, _ = load_decoder_checkpoint(decoder_path, device)
    compressor, decompressor, _ = load_compressor_checkpoint(compressor_path, device)
    decoder.eval()
    compressor.eval()
    decompressor.eval()
    h = cache["embeddings"][:limit].to(device)
    mask = cache["attention_mask"][:limit].to(device)
    lengths = cache["lengths"][:limit]
    with torch.no_grad():
        h_norm = normalizer.transform(h)
        logits = decoder(decompressor(compressor(h_norm, mask), mask), mask)
    return [decode_token_ids(row, length) for row, length in zip(logits.argmax(-1), lengths)]
