import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TrajectoryTransformer(nn.Module):
    def __init__(self, obs_dim=128, action_dim=32, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024):
        super().__init__()

        # 位置编码
        self.pos_encoder = PositionalEncoding(obs_dim)  # 使用obs_dim作为编码维度
        self.pos_encoder_action = PositionalEncoding(action_dim)  # 使用action_dim作为编码维度

        # Transformer编码器（用于动作掩码预测，双向注意力）
        self.action_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=obs_dim + action_dim, nhead=nhead,
                                       dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )

        # Transformer解码器（用于状态预测，单向注意力）
        self.state_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=obs_dim, nhead=nhead,
                                       dim_feedforward=dim_feedforward),
            num_layers=num_decoder_layers
        )

        # 输出层
        self.action_pred_head = nn.Linear(obs_dim + action_dim, action_dim)
        self.state_pred_head = nn.Linear(obs_dim, obs_dim)

    def forward(self, visual_feats, prev_actions):
        # visual_feats: [batch_size, seq_len, obs_dim]
        # prev_actions: [batch_size, seq_len, action_dim]

        batch_size, seq_len = visual_feats.shape[:2]

        # 添加位置编码
        obs_embedded = self.pos_encoder(visual_feats)
        action_embedded = self.pos_encoder_action(prev_actions)

        # 转置为transformer期望的格式 [seq_len, batch_size, dim]
        obs_embedded = obs_embedded.transpose(0, 1)
        action_embedded = action_embedded.transpose(0, 1)

        # 拼接观测和动作
        combined_features = torch.cat([obs_embedded, action_embedded], dim=-1)

        # 1. 动作掩码预测（双向注意力）
        action_hidden = self.action_transformer(combined_features)
        action_preds = self.action_pred_head(action_hidden.transpose(0, 1))

        # 2. 状态预测（单向注意力）
        # 生成因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(visual_feats.device)

        state_hidden = self.state_transformer(
            obs_embedded,  # target sequence
            combined_features,  # memory sequence
            tgt_mask=causal_mask
        )
        state_preds = self.state_pred_head(state_hidden.transpose(0, 1))

        return {
            'action_preds': action_preds,  # [batch_size, seq_len, action_dim]
            'state_preds': state_preds  # [batch_size, seq_len, obs_dim]
        }

    # 使用示例


if __name__ == "__main__":
    # 创建示例数据
    batch_size, seq_len = 5, 128
    obs_dim, action_dim = 128, 32

    visual_feats = torch.randn(batch_size, seq_len, obs_dim)
    prev_actions = torch.randn(batch_size, seq_len, action_dim)

    # 初始化模型
    model = TrajectoryTransformer()

    # 前向传播
    outputs = model(visual_feats, prev_actions)

    # 打印输出形状
    print("Action predictions shape:", outputs['action_preds'].shape)
    print("State predictions shape:", outputs['state_preds'].shape)