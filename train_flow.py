import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import CachedESM2Dataset, CompressedLatentDataset
from .normalizer import ESM2LatentNormalizer
from .simple_flow import build_simple_flow, simple_flow_loss
from .train_compressor import load_compressor_checkpoint
from .utils import ensure_parent_dir


def _make_compressed_cache(args, cache, device):
    if args.compressed_cache_path and os.path.exists(args.compressed_cache_path):
        return torch.load(args.compressed_cache_path, map_location="cpu")
    normalizer = ESM2LatentNormalizer.load(args.esm2_normalizer_path)
    compressor, _, _ = load_compressor_checkpoint(args.esm2_compressor_path, device)
    compressor.eval()
    compressor.requires_grad_(False)
    dataset = CachedESM2Dataset(cache)
    loader = DataLoader(dataset, batch_size=args.compressor_batch_size, shuffle=False)
    zs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Cache compressed z"):
            h = normalizer.transform(batch["embeddings"].to(device))
            mask = batch["attention_mask"].to(device)
            zs.append(compressor(h, attention_mask=mask).cpu())
    compressed = {
        "z": torch.cat(zs, dim=0),
        "attention_mask": cache["attention_mask"],
        "lengths": cache["lengths"],
        "compressed_dim": args.compressed_dim,
        "max_len": cache["max_len"],
    }
    if args.compressed_cache_path:
        ensure_parent_dir(args.compressed_cache_path)
        torch.save(compressed, args.compressed_cache_path)
        print(f"Saved compressed latent cache to {args.compressed_cache_path}")
    return compressed


def _run_epoch(model, loader, device, opt=None):
    is_train = opt is not None
    model.train(is_train)
    total = 0.0
    batches = 0
    for batch in tqdm(loader, leave=False):
        z = batch["z"].to(device)
        mask = batch["attention_mask"].to(device)
        if is_train:
            opt.zero_grad()
        loss = simple_flow_loss(model, z, mask)
        if is_train:
            loss.backward()
            opt.step()
        total += loss.item()
        batches += 1
    return total / max(1, batches)


def train_esm2_simple_flow(args):
    cache = torch.load(args.esm2_cache_path, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    compressed = _make_compressed_cache(args, cache, device)
    dataset = CompressedLatentDataset(compressed)
    val_size = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_set, val_set = (random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)) if val_size else (dataset, dataset))
    train_loader = DataLoader(train_set, batch_size=args.flow_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.flow_batch_size, shuffle=False)
    config = {
        "input_channels": args.compressed_dim,
        "out_channels": args.compressed_dim,
        "max_length": int(cache["max_len"]),
        "dropout": 0.1,
        "num_hidden_layers": 3,
        "num_attention_heads": 6,
    }
    model = build_simple_flow(args.compressed_dim, int(cache["max_len"]), config).to(device)
    opt = AdamW(model.parameters(), lr=args.flow_lr, weight_decay=args.flow_weight_decay)
    best = float("inf")
    ensure_parent_dir(args.esm2_flow_model_path)
    for epoch in range(args.flow_epoch):
        train_loss = _run_epoch(model, train_loader, device, opt)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, device)
        real_z = compressed["z"]
        print(
            f"Epoch {epoch}: train_flow_loss={train_loss:.5f} val_flow_loss={val_loss:.5f} "
            f"real_z_mean={real_z.mean().item():.4f} real_z_std={real_z.std().item():.4f}"
        )
        print("Rule: flow generation is meaningful only after decoder and compressor reconstruct well.")
        if val_loss < best:
            best = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": config, "best_val_loss": best}, args.esm2_flow_model_path)
    print(f"Saved best simple flow to {args.esm2_flow_model_path}")
