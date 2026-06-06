import torch

from .extract_embeddings import extract_esm2_embeddings
from .generate import generate_esm2_flow
from .normalizer import ESM2LatentNormalizer
from .simple_flow import build_simple_flow, sample_simple_flow
from .train_compressor import load_compressor_checkpoint, train_esm2_compressor
from .train_decoder import load_decoder_checkpoint, train_esm2_decoder
from .train_flow import train_esm2_simple_flow
from .utils import VOCAB, decode_token_ids, set_seed


ESM2_WORKS = {
    "ExtractESM2Embeddings",
    "TrainESM2Decoder",
    "TrainESM2Compressor",
    "TrainESM2SimpleFlow",
    "GenerateESM2Flow",
    "TestESM2Pipeline",
}


def test_esm2_pipeline(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cache = torch.load(args.esm2_cache_path, map_location="cpu")
    normalizer = ESM2LatentNormalizer.load(args.esm2_normalizer_path)
    decoder, _ = load_decoder_checkpoint(args.esm2_decoder_path, device)
    compressor, decompressor, compressor_ckpt = load_compressor_checkpoint(args.esm2_compressor_path, device)
    flow_ckpt = torch.load(args.esm2_flow_model_path, map_location=device)
    flow = build_simple_flow(args.compressed_dim, int(cache["max_len"]), flow_ckpt["config"]).to(device)
    flow.load_state_dict(flow_ckpt["model_state_dict"])
    decoder.eval()
    compressor.eval()
    decompressor.eval()
    flow.eval()

    batch_size = min(4, cache["embeddings"].shape[0])
    h = cache["embeddings"][:batch_size].to(device)
    mask = cache["attention_mask"][:batch_size].to(device)
    lengths = cache["lengths"][:batch_size]
    with torch.no_grad():
        h_norm = normalizer.transform(h)
        logits = decoder(h_norm, attention_mask=mask)
        z = compressor(h_norm, attention_mask=mask)
        h_rec = decompressor(z, attention_mask=mask)
        logits_rec = decoder(h_rec, attention_mask=mask)
        z_noise = torch.randn(batch_size, int(cache["max_len"]), compressor_ckpt["config"]["compressed_dim"], device=device)
        t = torch.rand(batch_size, device=device)
        v = flow(z_noise, t)
        z_gen = sample_simple_flow(flow, z_noise.shape, num_steps=args.flow_sample_steps, device=device)
        gen_logits = decoder(decompressor(z_gen, attention_mask=mask), attention_mask=mask)

    assert logits.shape == (batch_size, int(cache["max_len"]), len(VOCAB))
    assert z.shape == (batch_size, int(cache["max_len"]), compressor_ckpt["config"]["compressed_dim"])
    assert h_rec.shape == (batch_size, int(cache["max_len"]), int(cache["hidden_size"]))
    assert v.shape == z_noise.shape
    assert gen_logits.shape == (batch_size, int(cache["max_len"]), len(VOCAB))
    print(f"decoder logits shape: {tuple(logits.shape)}")
    print(f"z shape: {tuple(z.shape)}")
    print(f"h_rec shape: {tuple(h_rec.shape)}")
    print(f"flow output shape: {tuple(v.shape)}")
    print("Original reconstruction examples:")
    for row, length in zip(logits.argmax(-1), lengths):
        print(decode_token_ids(row, int(length)))
    print("Compression reconstruction examples:")
    for row, length in zip(logits_rec.argmax(-1), lengths):
        print(decode_token_ids(row, int(length)))
    print("Generated examples:")
    for row, length in zip(gen_logits.argmax(-1), lengths):
        print(decode_token_ids(row, int(length)))


def run_esm2_work(args):
    set_seed(args.seed)
    if args.work == "ExtractESM2Embeddings":
        extract_esm2_embeddings(args)
    elif args.work == "TrainESM2Decoder":
        train_esm2_decoder(args)
    elif args.work == "TrainESM2Compressor":
        train_esm2_compressor(args)
    elif args.work == "TrainESM2SimpleFlow":
        train_esm2_simple_flow(args)
    elif args.work == "GenerateESM2Flow":
        generate_esm2_flow(args)
    elif args.work == "TestESM2Pipeline":
        test_esm2_pipeline(args)
    else:
        raise ValueError(f"Unsupported ESM2 work mode: {args.work}")
