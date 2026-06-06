import torch
import torch.nn as nn


def _encoder(width, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class ESM2Compressor(nn.Module):
    def __init__(
        self,
        input_dim,
        compressed_dim=128,
        compressor_hidden_size=512,
        compressor_pre_layers=2,
        compressor_post_layers=2,
        compressor_heads=8,
        compressor_dropout=0.1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, compressor_hidden_size)
        self.pre = _encoder(compressor_hidden_size, compressor_heads, compressor_pre_layers, compressor_dropout)
        self.to_z = nn.Linear(compressor_hidden_size, compressed_dim)
        self.z_norm = nn.LayerNorm(compressed_dim)
        self.post = _encoder(compressed_dim, compressor_heads, compressor_post_layers, compressor_dropout)
        self.out = nn.Tanh()

    def forward(self, h, attention_mask=None):
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        x = self.input_proj(self.input_norm(h))
        x = self.pre(x, src_key_padding_mask=key_padding_mask)
        z = self.z_norm(self.to_z(x))
        z = self.post(z, src_key_padding_mask=key_padding_mask)
        z = self.out(z)
        if attention_mask is not None:
            z = z * attention_mask.unsqueeze(-1).to(z.dtype)
        return z


class ESM2Decompressor(nn.Module):
    def __init__(
        self,
        output_dim,
        compressed_dim=128,
        compressor_hidden_size=512,
        decompressor_pre_layers=2,
        decompressor_post_layers=2,
        compressor_heads=8,
        compressor_dropout=0.1,
    ):
        super().__init__()
        self.z_norm = nn.LayerNorm(compressed_dim)
        self.input_proj = nn.Linear(compressed_dim, compressor_hidden_size)
        self.pre = _encoder(compressor_hidden_size, compressor_heads, decompressor_pre_layers, compressor_dropout)
        self.to_h = nn.Linear(compressor_hidden_size, output_dim)
        self.h_norm = nn.LayerNorm(output_dim)
        self.post = _encoder(output_dim, compressor_heads, decompressor_post_layers, compressor_dropout)

    def forward(self, z, attention_mask=None):
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        x = self.input_proj(self.z_norm(z))
        x = self.pre(x, src_key_padding_mask=key_padding_mask)
        h = self.h_norm(self.to_h(x))
        h = self.post(h, src_key_padding_mask=key_padding_mask)
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)
        return h
