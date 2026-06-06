import csv
import os

import numpy as np
import torch

from .simple_flow import build_simple_flow, sample_simple_flow
from .train_compressor import load_compressor_checkpoint
from .train_decoder import load_decoder_checkpoint
from .utils import decode_token_ids, ensure_parent_dir


def _load_lengths(args):
    if args.length_distribution_path:
        data = torch.load(args.length_distribution_path, map_location="cpu")
        return data["lengths"] if isinstance(data, dict) and "lengths" in data else torch.as_tensor(data)
    if args.esm2_cache_path:
        return torch.load(args.esm2_cache_path, map_location="cpu")["lengths"]
    return torch.full((1000,), args.max_len, dtype=torch.long)


def generate_esm2_flow(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    decoder, _ = load_decoder_checkpoint(args.esm2_decoder_path, device)
    _, decompressor, compressor_ckpt = load_compressor_checkpoint(args.esm2_compressor_path, device)
    flow_ckpt = torch.load(args.esm2_flow_model_path, map_location=device)
    flow_model = build_simple_flow(args.compressed_dim, args.max_len, flow_ckpt["config"]).to(device)
    flow_model.load_state_dict(flow_ckpt["model_state_dict"])
    decoder.eval()
    decompressor.eval()
    flow_model.eval()

    compressed_dim = compressor_ckpt["config"].get("compressed_dim", args.compressed_dim)
    max_len = flow_ckpt["config"].get("max_length", args.max_len)
    lengths = _load_lengths(args).detach().cpu().numpy()
    sequences = []
    with torch.no_grad():
        for _ in range(args.Generate_times):
            z_gen = sample_simple_flow(
                flow_model,
                (args.Generate_batch_num, max_len, compressed_dim),
                num_steps=args.flow_sample_steps,
                device=device,
            )
            mask = torch.ones(args.Generate_batch_num, max_len, dtype=torch.long, device=device)
            logits = decoder(decompressor(z_gen, attention_mask=mask), attention_mask=mask)
            pred = logits.argmax(dim=-1)
            sampled_lengths = np.random.choice(lengths, size=args.Generate_batch_num, replace=True)
            for row, length in zip(pred, sampled_lengths):
                seq = decode_token_ids(row, int(length))
                if seq:
                    sequences.append(seq)

    ensure_parent_dir(args.Generate_save_path)
    with open(args.Generate_save_path, "w", encoding="utf-8") as handle:
        for seq in sequences:
            handle.write(f"{seq}\n")
    csv_path = os.path.splitext(args.Generate_save_path)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence"])
        writer.writerows([[seq] for seq in sequences])
    print(f"Saved {len(sequences)} generated sequences to {args.Generate_save_path} and {csv_path}")
