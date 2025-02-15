import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import random
from sub_models.attention_blocks import get_vector_mask
from sub_models.attention_blocks import PositionalEncoding1D, AttentionBlock, AttentionBlockKVCache


class StochasticTransformer(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary

        self.head = nn.Linear(feat_dim, stoch_dim)

    def forward(self, samples, action, mask):
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)

        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.action_mask_ratio = 0.2
        self.frame_mask_ratio = 0.2
        self.action_mask_token = nn.Parameter(torch.randn(1, 1, action_dim))
        self.frame_mask_token = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.frame_pred_head = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim),
            nn.GELU(),
            nn.Linear(stoch_dim, stoch_dim)
        )
        self.action_pred_head = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim),
            nn.GELU(),
            nn.Linear(stoch_dim, action_dim)
        )


        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary

    def forward(self, samples, action, mask):
        '''
        Normal forward pass
        '''
        # action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)

        return feats

    def mask_actions(self, actions):
        """随机掩码一部分动作"""
        batch_size, seq_len, _ = actions.shape
        masked_actions = actions.clone()
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=actions.device)

        # 为每个序列随机选择要掩码的位置
        for i in range(batch_size):
            num_masks = int(seq_len * self.action_mask_ratio)
            mask_indices = random.sample(range(seq_len), num_masks)
            mask[i, mask_indices] = True
            masked_actions[i][mask_indices] = self.action_mask_token.squeeze(0)

        return masked_actions, mask

    def mask_frames(self, samples):
        """随机掩码一部分图像帧"""
        batch_size, seq_len, feat_dim = samples.shape
        masked_samples = samples.clone()
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=samples.device)

        # 为每个序列随机选择要掩码的位置
        for i in range(batch_size):
            num_masks = int(seq_len * self.frame_mask_ratio)  # 使用单独的掩码比率
            mask_indices = random.sample(range(seq_len), num_masks)
            mask[i, mask_indices] = True
            # 使用专门用于图像帧的掩码token
            masked_samples[i][mask_indices] = self.frame_mask_token.squeeze(0)

        return masked_samples, mask

    def forward_with_masking(self, samples, action, attention_mask):
        '''
        Forward pass with both frame and action masking
        '''
        # 对图像帧和动作都进行掩码
        masked_samples, frame_mask = self.mask_frames(samples)
        masked_actions, action_mask = self.mask_actions(action)

        # 正常forward
        feats = self.stem(torch.cat([masked_samples, masked_actions], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        # 双向注意力
        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, attention_mask)

            # 分别预测被掩码的帧和动作
        frame_preds = self.frame_pred_head(feats)  # 新增的预测头
        action_preds = self.action_pred_head(feats)

        return frame_preds, action_preds, frame_mask, action_mask


    def reset_kv_cache_list(self, batch_size, dtype):
        '''
        Reset self.kv_cache_list
        '''
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device="cuda"))

    def forward_with_kv_cache(self, samples, action):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1]+1, samples.device)

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)

        return feats
