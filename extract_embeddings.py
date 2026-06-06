import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .dataset import build_internal_tokens, load_sequences
from .utils import VOCAB, ensure_parent_dir


def extract_esm2_embeddings(args):
    sequences = load_sequences(args.esm2_input_path, args.sequence_column, args.max_len)
    if not sequences:
        raise ValueError("No valid peptide sequences found after filtering.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.esm2_model_name)
    model = AutoModel.from_pretrained(args.esm2_model_name).to(device)
    model.eval()
    model.requires_grad_(False)

    token_ids, lengths, attention_mask = build_internal_tokens(sequences, args.max_len)
    embeddings = []
    for start in tqdm(range(0, len(sequences), args.esm2_batch_size), desc="Extract ESM2"):
        batch_sequences = sequences[start : start + args.esm2_batch_size]
        encoded = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding="max_length",
            truncation=False,
            max_length=args.max_len + 2,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        # ESM tokenizers add special tokens; keep peptide token positions only.
        batch_hidden = hidden[:, 1 : args.max_len + 1, :].detach().cpu()
        embeddings.append(batch_hidden)

    embeddings = torch.cat(embeddings, dim=0)
    hidden_size = int(getattr(model.config, "hidden_size", embeddings.shape[-1]))
    cache = {
        "embeddings": embeddings,
        "attention_mask": attention_mask,
        "token_ids": token_ids,
        "lengths": lengths,
        "sequences": sequences,
        "esm2_model_name": args.esm2_model_name,
        "hidden_size": hidden_size,
        "max_len": args.max_len,
        "vocab": VOCAB,
    }
    ensure_parent_dir(args.esm2_cache_path)
    torch.save(cache, args.esm2_cache_path)
    print(f"Saved {len(sequences)} ESM2 embeddings to {args.esm2_cache_path}")
    print(f"hidden_size={hidden_size}, max_len={args.max_len}, model={args.esm2_model_name}")
