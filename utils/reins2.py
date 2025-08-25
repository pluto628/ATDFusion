import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul


class LoRAReins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        token_length: int = 100,
        lora_dim: int = 16,
        use_softmax: bool = True,
        scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.token_length = token_length
        self.lora_dim = lora_dim
        self.use_softmax = use_softmax
        self.scale_init = scale_init

        self.create_model()

    def create_model(self):
        # LoRA decomposition
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )

        val = math.sqrt(
            6.0 / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

        self.token_length_predictor = nn.Sequential(
            nn.Linear(self.embed_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.rank_predictor = nn.Sequential(
            nn.Linear(self.embed_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Token to feature
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)

        self.gate_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def get_tokens(self, layer: int, feat_sample: torch.Tensor) -> tuple[torch.Tensor, list[int]]:

        B = feat_sample.shape[1]
        embed = feat_sample.mean(dim=0)  # [B, C]

        token_ratios = self.token_length_predictor(embed).squeeze(-1)  # [B]
        token_ratios = torch.sigmoid(token_ratios)
        token_counts = (token_ratios * self.token_length).long().clamp(min=1, max=self.token_length)  # [B]
        max_token_len = token_counts.max().item()

        rank_ratios = self.rank_predictor(embed).squeeze(-1)
        ranks = (rank_ratios * self.lora_dim).long().clamp(min=1, max=self.lora_dim)

        A_all = self.learnable_tokens_a[layer]  # [T, r]
        B_all = self.learnable_tokens_b[layer]  # [r, C]

        tokens = []
        for i in range(B):
            t_len = token_counts[i]
            r = ranks[i]

            A = A_all[:t_len, :r]
            B = B_all[:r, :]
            tok = A @ B  # [t_len, C]

            pad = torch.zeros(max_token_len - t_len, tok.shape[1], device=tok.device)
            tok = torch.cat([tok, pad], dim=0)
            tokens.append(tok)

        tokens = torch.stack(tokens, dim=0)  # [B, max_token_len, C]

        # if self.training:
        #     print("Layer", layer, "token_counts:", token_counts)

        return tokens, token_counts.tolist()

    def forward_delta_feat(self, feats: torch.Tensor, tokens: torch.Tensor, token_counts: list[int]) -> torch.Tensor:

        N, B, C = feats.shape
        T = tokens.shape[1]

        attn = torch.einsum("nbc,btc->nbt", feats, tokens)

        if self.use_softmax:
            attn = attn / (C ** 0.5)

            # Mask padding token
            mask = torch.zeros(B, T, device=feats.device)
            for i, count in enumerate(token_counts):
                mask[i, :count] = 1
            mask = mask.unsqueeze(0)  # [1, B, T]
            attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)

        # token -> feat
        tokens_proj = self.mlp_token2feat(tokens)  # [B, T, C]
        delta_feat = torch.einsum("nbt,btc->nbc", attn, tokens_proj)

        # Gating
        gate = self.gate_layer(feats)
        delta_feat = delta_feat * gate
        return delta_feat

    def forward(
        self,
        feats: torch.Tensor,
        layer: int,
        batch_first=False,
        has_cls_token=True,
    ) -> torch.Tensor:
        # feats: [B, N, C] or [N, B, C]
        if batch_first:
            feats = feats.permute(1, 0, 2)  # -> [N, B, C]
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        tokens, token_counts = self.get_tokens(layer, feat_sample=feats)
        delta_feat = self.forward_delta_feat(feats, tokens, token_counts)

        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)

        return feats
