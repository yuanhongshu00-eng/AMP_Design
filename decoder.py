import torch.nn as nn


class ESM2LatentDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        decoder_hidden_size=512,
        decoder_layers=4,
        decoder_heads=8,
        decoder_dropout=0.1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, decoder_hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_hidden_size,
            nhead=decoder_heads,
            dropout=decoder_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=decoder_layers)
        self.output_proj = nn.Linear(decoder_hidden_size, vocab_size)

    def forward(self, h, attention_mask=None):
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        x = self.input_proj(self.input_norm(h))
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.output_proj(x)
