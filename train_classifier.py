"""
AMPClassifier Training Script (Updated for FASTA & Optimized)
===========================================================
基于 classifier.py 中的 AMPClassifier 模型进行训练。
已修改：
1. 替换 CSV 加载逻辑，直接解析 `data_positive.fasta` 和 `data_negative.fasta`。
2. max_len 从 128 缩短到 64，减少无意义的 padding 计算开销。
3. 保持了你原有的渐进式解冻策略 (freeze_epochs=5) 和类权重平衡 (pos_weight)。
"""

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import EsmTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score, recall_score
)
from tqdm import tqdm

# 从外部导入用户提供的模型
from classifier import AMPClassifier


# ------------------------------------------------------------------
# 0. 配置（集中管理，无 argparse）
# ------------------------------------------------------------------

@dataclass
class Config:
    # 数据路径 (已修改为直接读取 FASTA)
    pos_fasta: str = "Data/data_positive.fasta"
    neg_fasta: str = "Data/data_negative.fasta"
    max_len: int = 64  # 优化点：缩短最大截断长度

    # 模型参数（与 classifier.py 的 AMPClassifier 对齐）
    esm_model_name: str = "facebook/esm2_t12_35M_UR50D"
    freeze_esm2: bool = True          # 前 freeze_epochs 轮保持冻结
    proj_dim: int = 256
    cnn_filters: int = 64
    lstm_hidden: int = 128
    dropout: float = 0.3

    # 训练参数
    batch_size: int = 32
    epochs: int = 50
    lr: float = 2e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.1         # warmup 占总步数比例
    grad_clip: float = 1.0
    pos_weight: float = 0.0           # 0 表示自动按 负:正 计算
    freeze_epochs: int = 5            # 前 N 轮冻结 ESM2，之后解冻最后 4 层

    # 系统
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 输出与早停
    save_dir: str = "./checkpoints"
    patience: int = 5                 # 早停容忍轮数
    monitor_metric: str = "val_aupr"  # 早停监控指标


# ------------------------------------------------------------------
# 1. 数据集 (新增 FASTA 解析逻辑)
# ------------------------------------------------------------------

def read_fasta(file_path: str, label: int) -> List[Tuple[str, int]]:
    """解析 FASTA 文件，返回 (sequence, label) 列表"""
    pairs = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    pairs.append((seq, label))
                    seq = ""
            elif line:
                seq += line
        if seq:  # 保存最后一条
            pairs.append((seq, label))
    return pairs


class AMPDataset(Dataset):
    """抗菌肽二分类数据集"""
    def __init__(self, pairs: List[Tuple[str, int]], tokenizer, max_len: int):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        seq, label = self.pairs[idx]
        enc = self.tok(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32)
        }


def build_dataloaders(cfg: Config):
    """构建训练/验证 DataLoader，并自动计算 pos_weight"""
    tokenizer = EsmTokenizer.from_pretrained(cfg.esm_model_name)

    # 1. 读取正负样本 FASTA
    pos_pairs = read_fasta(cfg.pos_fasta, 1)
    neg_pairs = read_fasta(cfg.neg_fasta, 0)

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)

    # 2. 划分 90% 训练集, 10% 验证集
    split = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split]
    val_pairs = all_pairs[split:]

    # 3. 统计正负样本比例，计算类别权重
    labels = [l for _, l in train_pairs]
    pos = sum(labels)
    neg = len(labels) - pos
    ratio = neg / max(pos, 1)
    print(f"[Data] Train={len(train_pairs)}  Val={len(val_pairs)}  "
          f"Pos={pos} Neg={neg}  Ratio=1:{ratio:.1f}")

    if cfg.pos_weight <= 0:
        cfg.pos_weight = ratio
        print(f"[Auto] pos_weight = {cfg.pos_weight:.2f}")

    train_loader = DataLoader(
        AMPDataset(train_pairs, tokenizer, cfg.max_len),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        AMPDataset(val_pairs, tokenizer, cfg.max_len),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader, tokenizer


# ------------------------------------------------------------------
# 2. 训练与验证的单步逻辑
# ------------------------------------------------------------------

def _compute_metrics(
    losses: List[float], probs: List[float], labels: List[float], prefix: str
) -> Dict[str, float]:
    """计算不平衡敏感指标（AUROC / AUPR / MCC / F1 / Recall）"""
    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        f"{prefix}_loss": float(np.mean(losses)),
        f"{prefix}_acc": float((y_pred == y_true).mean()),
        f"{prefix}_auroc": float(roc_auc_score(y_true, y_prob)),
        f"{prefix}_aupr": float(average_precision_score(y_true, y_prob)),
        f"{prefix}_mcc": float(matthews_corrcoef(y_true, y_pred)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    scaler: GradScaler,
    cfg: Config
) -> Dict[str, float]:
    """训练一个 epoch，使用混合精度加速"""
    model.train()
    losses, all_probs, all_labels = [], [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        mask = batch["attention_mask"].to(cfg.device, non_blocking=True)
        y = batch["labels"].to(cfg.device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()

        with autocast():
            # 现在模型返回的是 logits
            logits, _ = model(ids, mask)
            # BCEWithLogitsLoss 直接吃 logits
            loss = criterion(logits, y)

        # 【新增】为了计算 AUROC、AUPR 等指标，我们需要手动把 logits 变成概率
        probs = torch.sigmoid(logits)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        all_probs.extend(probs.detach().cpu().numpy().ravel())
        all_labels.extend(y.detach().cpu().numpy().ravel())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return _compute_metrics(losses, all_probs, all_labels, prefix="train")


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: Config
) -> Dict[str, float]:
    """验证一个 epoch"""
    model.eval()
    losses, all_probs, all_labels = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        mask = batch["attention_mask"].to(cfg.device, non_blocking=True)
        y = batch["labels"].to(cfg.device, non_blocking=True).unsqueeze(1)

        with autocast():
            logits, _ = model(ids, mask)
            loss = criterion(logits, y)

        # 【新增】手动算概率供指标评估使用
        probs = torch.sigmoid(logits)

        losses.append(loss.item())
        all_probs.extend(probs.cpu().numpy().ravel())
        all_labels.extend(y.cpu().numpy().ravel())


    return _compute_metrics(losses, all_probs, all_labels, prefix="val")


# ------------------------------------------------------------------
# 3. 早停（内置，零外部依赖）
# ------------------------------------------------------------------

class EarlyStopping:
    """基于验证指标的单点早停，监控 AUPR（对不平衡数据最敏感）"""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False


# ------------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 数据
    train_loader, val_loader, tokenizer = build_dataloaders(cfg)

    # 模型（直接导入用户的 AMPClassifier）
    model = AMPClassifier(
        esm_model_name=cfg.esm_model_name,
        freeze_esm2=cfg.freeze_esm2,
        proj_dim=cfg.proj_dim,
        cnn_filters=cfg.cnn_filters,
        lstm_hidden=cfg.lstm_hidden,
        dropout=cfg.dropout
    ).to(cfg.device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total={total:,}  Trainable={trainable:,}  ({trainable/total*100:.1f}%)")

    # 损失: Weighted BCE
    pos_w = torch.tensor([cfg.pos_weight]).to(cfg.device)
    base_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def weighted_bce_loss(probs, targets):
        weights = torch.where(targets == 1, pos_w, torch.ones_like(pos_w))
        loss = base_criterion(probs, targets)
        return (loss * weights).mean()

    criterion = weighted_bce_loss

    # 优化器（分组学习率：ESM2 微调更慢）
    esm_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": esm_params, "lr": cfg.lr * 0.1},
        {"params": other_params, "lr": cfg.lr}
    ], weight_decay=cfg.weight_decay)

    # 学习率调度: cosine with warmup
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 混合精度
    scaler = GradScaler()

    # 早停
    early_stop = EarlyStopping(patience=cfg.patience)
    best_path = os.path.join(cfg.save_dir, "best_model.pt")
    history: List[Dict] = []

    print(f"[Train] device={cfg.device}  epochs={cfg.epochs}  steps={total_steps}  warmup={warmup_steps}")
    print("-" * 70)

    for epoch in range(cfg.epochs):
        # 阶段性解冻 ESM2 最后 4 层
        if epoch == cfg.freeze_epochs:
            print(f">>> Epoch {epoch}: 解冻 ESM2 最后 4 层...")
            for name, p in model.backbone.esm2.named_parameters():
                if "layer." in name:
                    try:
                        idx = int(name.split("layer.")[1].split(".")[0])
                        if idx >= 8:
                            p.requires_grad = True
                    except Exception:
                        pass
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"       Trainable now: {trainable:,}")

        # 训练 & 验证
        train_m = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, cfg)
        val_m = eval_epoch(model, val_loader, criterion, cfg)

        # 日志
        log_line = (
            f"Epoch {epoch:02d} | "
            f"loss={train_m['train_loss']:.4f}/{val_m['val_loss']:.4f} | "
            f"AUROC={train_m['train_auroc']:.4f}/{val_m['val_auroc']:.4f} | "
            f"AUPR={train_m['train_aupr']:.4f}/{val_m['val_aupr']:.4f} | "
            f"MCC={val_m['val_mcc']:.4f} | Recall={val_m['val_recall']:.4f}"
        )
        print(log_line)

        merged = {**train_m, **val_m, "epoch": epoch}
        history.append(merged)

        # 早停 & 保存最优模型
        improved = early_stop(val_m["val_aupr"])
        if improved:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "metrics": merged
            }, best_path)
            print(f"       -> Saved best (val_aupr={val_m['val_aupr']:.4f})")

        if early_stop.stop:
            print(f"!!! Early stopping at epoch {epoch}")
            break

    # 导出训练历史
    with open(os.path.join(cfg.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Done] Best model: {best_path}  History: {cfg.save_dir}/history.json")


if __name__ == "__main__":
    main()