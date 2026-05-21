import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel

class ESM2Backbone(nn.Module):
    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D", freeze=False):
        super().__init__()
        self.esm2 = EsmModel.from_pretrained(model_name)
        self.hidden_dim = self.esm2.config.hidden_size  # 35M: 480, 150M: 640

        if freeze:
            for param in self.esm2.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # 输入: (batch_size, seq_len) 的 token ids
        # 输出: (batch_size, seq_len, hidden_dim) 最后一层 hidden states
        outputs = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (B, L, H)


class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 输入: (B, L, 480)  —— ESM2 原始输出
        # 输出: (B, L, 256)  —— 降维后，供 CNN/LSTM 共享
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class InceptionCNN(nn.Module):
    def __init__(self, input_dim=256, num_filters=64):
        super().__init__()
        # 4 个尺度并行：1-mer / 3-mer / 5-mer / 7-mer
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, num_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, num_filters, kernel_size=7, padding=3)

        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.bn5 = nn.BatchNorm1d(num_filters)
        self.bn7 = nn.BatchNorm1d(num_filters)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 输入: (B, L, 256)
        x = x.transpose(1, 2)  # Conv1d 需要 (B, C, L)

        out1 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, L)
        out3 = self.relu(self.bn3(self.conv3(x)))  # (B, 64, L)
        out5 = self.relu(self.bn5(self.conv5(x)))  # (B, 64, L)
        out7 = self.relu(self.bn7(self.conv7(x)))  # (B, 64, L)

        out = torch.cat([out1, out3, out5, out7], dim=1)  # (B, 256, L)
        out = torch.max(out, dim=2)[0]  # Global Max Pooling → (B, 256)
        out = self.dropout(out)
        return out


class BiLSTMBranch(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0  # num_layers=1 时 dropout 无效
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # 输入: (B, L, 256)
        lstm_out, _ = self.lstm(x)  # (B, L, 256)  双向 128*2

        # 用 attention_mask 做加权平均，彻底排除 padding 干扰
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            sum_out = (lstm_out * mask).sum(dim=1)  # (B, 256)
            mean_out = sum_out / mask.sum(dim=1).clamp(min=1)
        else:
            mean_out = lstm_out.mean(dim=1)  # (B, 256)

        return self.dropout(mean_out)


class SEFusion(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        # 瓶颈结构: input_dim -> input_dim//4 -> input_dim
        self.fc1 = nn.Linear(input_dim, input_dim // reduction)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, local_feat, global_feat):
        # local_feat: (B, 256)  |  global_feat: (B, 256)
        fused = torch.cat([local_feat, global_feat], dim=-1)  # (B, 512)

        # 学习每个维度的权重
        w = self.sigmoid(self.fc2(self.relu(self.fc1(fused))))  # (B, 512)

        # 重标定：重要特征放大，冗余特征抑制
        refined = fused * w  # (B, 512)
        return self.dropout(self.layer_norm(refined))


class ClassificationHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=1, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc_out = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 第一层: 512 -> 128
        out = self.dropout(self.relu(self.bn1(self.fc1(x))))

        # 第二层 + 残差: 128 -> 128
        residual = out
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))
        out = out + residual  # 残差连接

        return self.fc_out(out)  # (B, 1) logits


class AMPClassifier(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t12_35M_UR50D",
                 freeze_esm2=False, proj_dim=256,
                 cnn_filters=64, lstm_hidden=128, dropout=0.3):
        super().__init__()

        # 1. 主干
        self.backbone = ESM2Backbone(esm_model_name, freeze=freeze_esm2)
        esm_dim = self.backbone.hidden_dim

        # 2. 投影
        self.projection = FeatureProjection(esm_dim, proj_dim)

        # 3. 双分支
        self.cnn_branch = InceptionCNN(proj_dim, cnn_filters)
        self.lstm_branch = BiLSTMBranch(proj_dim, lstm_hidden, dropout=dropout)

        # 4. 融合
        local_dim = 4 * cnn_filters  # 256
        global_dim = 2 * lstm_hidden  # 256
        self.fusion = SEFusion(local_dim + global_dim)

        # 5. 分类
        self.classifier = ClassificationHead(local_dim + global_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        features = {}

        # 数据流 (带维度)
        esm_feat = self.backbone(input_ids, attention_mask)
        proj_feat = self.projection(esm_feat)

        local_feat = self.cnn_branch(proj_feat)
        global_feat = self.lstm_branch(proj_feat, attention_mask)

        fused = self.fusion(local_feat, global_feat)
        logits = self.classifier(fused)  # (B,1)

        # 【修改点】删除原有的 probs = self.sigmoid(logits)
        # 直接返回 logits
        return logits, features