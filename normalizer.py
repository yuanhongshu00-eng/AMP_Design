import torch


class ESM2LatentNormalizer:
    def __init__(self, mean=None, std=None, clip_value=5.0, use_tanh_smoothing=False, eps=1e-6):
        self.mean = mean
        self.std = std
        self.clip_value = clip_value
        self.use_tanh_smoothing = use_tanh_smoothing
        self.eps = eps

    def fit(self, embeddings, attention_mask):
        mask = attention_mask.bool()
        valid = embeddings[mask]
        self.mean = valid.mean(dim=0)
        self.std = valid.std(dim=0).clamp_min(self.eps)
        return self

    def transform(self, embeddings):
        mean = self.mean.to(embeddings.device)
        std = self.std.to(embeddings.device)
        h = (embeddings - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + self.eps)
        h = h.clamp(-self.clip_value, self.clip_value)
        if self.use_tanh_smoothing:
            h = torch.tanh(h)
        return h

    def save(self, path):
        torch.save(
            {
                "mean": self.mean.detach().cpu(),
                "std": self.std.detach().cpu(),
                "clip_value": self.clip_value,
                "use_tanh_smoothing": self.use_tanh_smoothing,
                "eps": self.eps,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        data = torch.load(path, map_location=map_location)
        return cls(
            mean=data["mean"],
            std=data["std"],
            clip_value=data.get("clip_value", 5.0),
            use_tanh_smoothing=data.get("use_tanh_smoothing", False),
            eps=data.get("eps", 1e-6),
        )
