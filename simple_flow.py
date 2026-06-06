import torch

from src.LatentFlowMatching import FlowMatchingModel, LatentFlowMatching
from .utils import masked_mean


def build_simple_flow(compressed_dim=128, max_len=50, config=None):
    config = config or {}
    return FlowMatchingModel(
        input_channels=config.get("input_channels", compressed_dim),
        out_channels=config.get("out_channels", compressed_dim),
        max_length=config.get("max_length", max_len),
        dropout=config.get("dropout", 0.1),
        num_hidden_layers=config.get("num_hidden_layers", 3),
        num_attention_heads=config.get("num_attention_heads", 6),
    )


def simple_flow_loss(model, z, attention_mask):
    x0 = torch.randn_like(z)
    t = torch.rand(z.shape[0], device=z.device)
    x_t = (1.0 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * z
    v_target = z - x0
    v_pred = model(x_t, t)
    assert v_pred.shape == v_target.shape == z.shape
    return masked_mean((v_pred - v_target) ** 2, attention_mask)


def sample_simple_flow(model, shape, num_steps=100, device=None):
    return LatentFlowMatching().sample(model, shape, num_steps=num_steps, device=device)
