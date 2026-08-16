"""MolmoAct2 V3 goal-pose prior modules adapted to FLUX.2 ImageWAM.

Stage1 encodes an oracle future pose into 8 FLUX hidden tokens.
Stage2 infers 100 recurrent latents (8 pose + 92 context) from language/state
and current-image tokens, then steers ActionDiT through synthetic K/V.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

GOAL_PRIOR_NUM_GOAL_TOKENS = 8
GOAL_PRIOR_NUM_LATENTS = 100
GOAL_PRIOR_NUM_POSE_TOKENS = 8
GOAL_PRIOR_NUM_GROUPS = 5
GOAL_PRIOR_LATENT_DIM = 768
GOAL_PRIOR_INNER_DIM = 512
GOAL_PRIOR_FFN_RATIO = 4.0
GOAL_PRIOR_NUM_HEADS = 8
GOAL_PRIOR_DROPOUT = 0.0
GOAL_PRIOR_POSE_LOSS_WEIGHT = 0.3
GOAL_PRIOR_SYNTHETIC_TIME_VALUE = 3.0

STAGE1_ONLY_CHECKPOINT_PREFIXES = ("goal_pose_encoder.",)
STAGE2_ONLY_CHECKPOINT_PREFIXES = (
    "semantic_visual_aggregator.",
    "semantic_visual_pose_norm.",
    "semantic_visual_pose_decoder.",
)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_f = x.float()
        rrms = torch.rsqrt(torch.mean(x_f**2, dim=-1, keepdim=True) + self.eps)
        return (x_f * rrms).to(dtype=x_dtype) * self.scale


class GoalPoseEncoder(nn.Module):
    """Maps a normalized goal pose vector to K FLUX hidden tokens."""

    def __init__(
        self,
        pose_dim: int,
        num_tokens: int = GOAL_PRIOR_NUM_GOAL_TOKENS,
        hidden_size: int = 3072,
        inner_dim: int = GOAL_PRIOR_INNER_DIM,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.pose_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, self.num_tokens * self.hidden_size),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 2:
            raise ValueError(f"`pose` must be [B, pose_dim], got {tuple(pose.shape)}")
        if pose.shape[1] != self.pose_dim:
            raise ValueError(f"`pose` last dim must be {self.pose_dim}, got {pose.shape[1]}")
        return self.net(pose).view(pose.shape[0], self.num_tokens, self.hidden_size)


class GoalPoseDecoder(nn.Module):
    """Reconstructs a normalized goal pose from concatenated pose latents."""

    def __init__(
        self,
        num_tokens: int = GOAL_PRIOR_NUM_POSE_TOKENS,
        hidden_size: int = GOAL_PRIOR_LATENT_DIM,
        pose_dim: int = 8,
        inner_dim: int = GOAL_PRIOR_INNER_DIM,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.pose_dim = int(pose_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_tokens * self.hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, self.pose_dim),
        )

    def forward(self, goal_hidden: torch.Tensor) -> torch.Tensor:
        if goal_hidden.ndim != 3:
            raise ValueError(f"`goal_hidden` must be [B, K, D], got {tuple(goal_hidden.shape)}")
        if goal_hidden.shape[1] != self.num_tokens or goal_hidden.shape[2] != self.hidden_size:
            raise ValueError(
                f"`goal_hidden` must be [B, {self.num_tokens}, {self.hidden_size}], got {tuple(goal_hidden.shape)}"
            )
        return self.net(goal_hidden.reshape(goal_hidden.shape[0], -1))


class _SemanticVisualSelfAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, latent_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(tokens)
        update, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.dropout(update)
        return tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))


class _SemanticVisualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, latent_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B, S, D], got {tuple(context.shape)}")
        if valid_mask.ndim != 2 or valid_mask.shape != context.shape[:2]:
            raise ValueError(
                f"`valid_mask` must be [B, S]={tuple(context.shape[:2])}, got {tuple(valid_mask.shape)}"
            )
        key_padding_mask = ~valid_mask.to(dtype=torch.bool)
        # MultiheadAttention forbids fully-masked keys; keep one dummy position.
        fully_masked = key_padding_mask.all(dim=1)
        if bool(fully_masked.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_masked, 0] = False
        context_normed = self.context_norm(context)
        update, _ = self.cross_attn(
            self.query_norm(queries),
            context_normed,
            context_normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class SemanticVisualAggregatorGroup(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        kv_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
        attn_head_dim: int,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.kv_dim = int(kv_dim)
        self.attn_head_dim = int(attn_head_dim)
        self.num_kv_heads = self.kv_dim // self.attn_head_dim
        if self.num_kv_heads * self.attn_head_dim != self.kv_dim:
            raise ValueError(f"`kv_dim` ({kv_dim}) must be divisible by attn_head_dim ({attn_head_dim}).")
        self.self_block = _SemanticVisualSelfAttentionBlock(latent_dim, num_heads, ffn_ratio, dropout)
        self.semantic_block = _SemanticVisualCrossAttentionBlock(
            latent_dim, context_dim, num_heads, ffn_ratio, dropout
        )
        self.visual_block = _SemanticVisualCrossAttentionBlock(
            latent_dim, context_dim, num_heads, ffn_ratio, dropout
        )
        self.to_key = nn.Linear(latent_dim, kv_dim, bias=False)
        self.to_value = nn.Linear(latent_dim, kv_dim, bias=False)
        self.key_norm = _RMSNorm(self.attn_head_dim)

    def forward(
        self,
        queries: torch.Tensor,
        semantic_hidden: torch.Tensor,
        visual_hidden: torch.Tensor,
        *,
        semantic_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.self_block(queries)
        queries = self.semantic_block(queries, semantic_hidden, semantic_mask)
        return self.visual_block(queries, visual_hidden, image_mask)

    def project_kv(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.to_key(tokens)
        value = self.to_value(tokens)
        batch, seq_len, _ = key.shape
        key = key.view(batch, seq_len, self.num_kv_heads, self.attn_head_dim)
        key = self.key_norm(key)
        key = key.reshape(batch, seq_len, self.kv_dim)
        return key, value


class SemanticVisualAggregator(nn.Module):
    """Recurrent 100-token V3 aggregator with depth-grouped parameters."""

    def __init__(
        self,
        num_tokens: int = GOAL_PRIOR_NUM_LATENTS,
        latent_dim: int = GOAL_PRIOR_LATENT_DIM,
        context_dim: int = 3072,
        kv_dim: int = 3072,
        attn_head_dim: int = 128,
        num_layer_groups: int = GOAL_PRIOR_NUM_GROUPS,
        num_heads: int = GOAL_PRIOR_NUM_HEADS,
        ffn_ratio: float = GOAL_PRIOR_FFN_RATIO,
        dropout: float = GOAL_PRIOR_DROPOUT,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.kv_dim = int(kv_dim)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layer_groups = int(num_layer_groups)
        if self.num_layer_groups < 1:
            raise ValueError(f"`num_layer_groups` must be >= 1, got {self.num_layer_groups}")
        queries = torch.empty(self.num_tokens, self.latent_dim)
        nn.init.trunc_normal_(queries, std=0.02)
        self.queries = nn.Parameter(queries)
        self.groups = nn.ModuleList(
            [
                SemanticVisualAggregatorGroup(
                    latent_dim=self.latent_dim,
                    context_dim=self.context_dim,
                    kv_dim=self.kv_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    attn_head_dim=self.attn_head_dim,
                )
                for _ in range(self.num_layer_groups)
            ]
        )

    def initial_queries(self, batch_size: int) -> torch.Tensor:
        return self.queries.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()

    def layer_group_index(self, layer_idx: int, num_layers: int) -> int:
        return layer_group_index(layer_idx, num_layers, self.num_layer_groups)

    def forward_layer(
        self,
        queries: torch.Tensor,
        semantic_hidden: torch.Tensor,
        visual_hidden: torch.Tensor,
        *,
        semantic_mask: torch.Tensor,
        image_mask: torch.Tensor,
        layer_idx: int,
        num_layers: int,
    ) -> torch.Tensor:
        group = self.groups[self.layer_group_index(layer_idx, num_layers)]
        return group(
            queries,
            semantic_hidden,
            visual_hidden,
            semantic_mask=semantic_mask,
            image_mask=image_mask,
        )

    def project_kv(
        self,
        tokens: torch.Tensor,
        *,
        layer_idx: int,
        num_layers: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group = self.groups[self.layer_group_index(layer_idx, num_layers)]
        return group.project_kv(tokens)


def layer_group_index(layer_idx: int, num_layers: int, num_layer_groups: int) -> int:
    if int(num_layers) <= 0:
        raise ValueError(f"`num_layers` must be positive, got {num_layers}")
    if int(num_layers) % int(num_layer_groups) != 0:
        raise ValueError(
            f"`num_layers` ({num_layers}) must be divisible by num_layer_groups ({num_layer_groups})"
        )
    if not 0 <= int(layer_idx) < int(num_layers):
        raise ValueError(f"`layer_idx` must be in [0, {num_layers}), got {layer_idx}")
    return int(layer_idx) // (int(num_layers) // int(num_layer_groups))


def extract_goal_pose_from_proprio(proprio: torch.Tensor, proprio_is_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the last observation as goal before aligning proprio with the action horizon."""
    if proprio.ndim != 2:
        raise ValueError(f"`proprio` must be [num_obs, dim], got {tuple(proprio.shape)}")
    if proprio.shape[0] < 2:
        raise ValueError(f"`proprio` must contain current state and future goal, got {tuple(proprio.shape)}")
    if proprio_is_pad.ndim != 1 or proprio_is_pad.shape[0] != proprio.shape[0]:
        raise ValueError(
            f"`proprio_is_pad` must be [{proprio.shape[0]}], got {tuple(proprio_is_pad.shape)}"
        )
    return proprio[-1].clone(), proprio_is_pad[-1].clone()


def build_synthetic_token_ids(
    batch_size: int,
    seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    time_value: float = GOAL_PRIOR_SYNTHETIC_TIME_VALUE,
) -> torch.Tensor:
    ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
    ids[..., 0] = float(time_value)
    ids[..., 1] = torch.arange(seq_len, device=device, dtype=dtype)[None, :]
    return ids


def build_stage2_action_attention_mask(
    batch_size: int,
    txt_len: int,
    synthetic_len: int,
    action_len: int,
    device: torch.device,
    text_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Action queries attend to valid text/state, all synthetic tokens, and action self."""
    key_len = int(txt_len) + int(synthetic_len) + int(action_len)
    mask = torch.zeros(batch_size, action_len, key_len, dtype=torch.bool, device=device)
    mask[:, :, :txt_len] = True
    mask[:, :, txt_len : txt_len + synthetic_len] = True
    mask[:, :, txt_len + synthetic_len :] = True
    if text_attention_mask is not None:
        if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
            raise ValueError(
                "`text_attention_mask` must be [B, txt_len], "
                f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
            )
        mask[:, :, :txt_len] &= text_attention_mask.to(device=device, dtype=torch.bool)[:, None, :]
    return mask


def build_concatenated_hidden(
    semantic: torch.Tensor,
    visual: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([semantic, visual], dim=1)


def build_concatenated_masks(
    semantic_mask: torch.Tensor,
    visual_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic_full = torch.cat(
        [semantic_mask, torch.zeros_like(visual_mask, dtype=torch.bool)],
        dim=1,
    )
    visual_full = torch.cat(
        [torch.zeros_like(semantic_mask, dtype=torch.bool), visual_mask],
        dim=1,
    )
    return semantic_full, visual_full


def compute_pose_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_pad: Optional[torch.Tensor] = None,
    dim_is_pad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pose pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    per_dim = F.mse_loss(pred.float(), target.float(), reduction="none")
    if dim_is_pad is not None:
        dim_valid = (~dim_is_pad).to(device=per_dim.device, dtype=per_dim.dtype)
        if dim_valid.ndim == 1:
            dim_valid = dim_valid.unsqueeze(0).expand(per_dim.shape[0], -1)
        dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0)
        per_sample = (per_dim * dim_valid).sum(dim=1) / dim_valid_sum
    else:
        per_sample = per_dim.mean(dim=1)
    if is_pad is None:
        return per_sample.mean()
    valid = (~is_pad.view(-1)).to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)


def _starts_with_any(key: str, prefixes: Sequence[str]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def validate_goal_prior_checkpoint_keys(
    *,
    current_stage: str,
    payload_stage: Optional[str],
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    bridge_from_stage1: bool = False,
) -> None:
    """Fail-closed key check for Stage1/Stage2 checkpoints."""
    current_stage = str(current_stage)
    missing = list(missing_keys)
    unexpected = list(unexpected_keys)
    if current_stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported goal-prior stage {current_stage!r}")

    if current_stage == "stage1":
        if missing:
            raise RuntimeError(f"Stage1 checkpoint is missing keys: {missing[:20]}")
        if unexpected:
            raise RuntimeError(f"Stage1 checkpoint has unexpected keys: {unexpected[:20]}")
        return

    allowed_missing = STAGE2_ONLY_CHECKPOINT_PREFIXES if bridge_from_stage1 else ()
    allowed_unexpected = STAGE1_ONLY_CHECKPOINT_PREFIXES if bridge_from_stage1 else ()
    bad_missing = [key for key in missing if not _starts_with_any(key, allowed_missing)]
    bad_unexpected = [key for key in unexpected if not _starts_with_any(key, allowed_unexpected)]
    if bad_missing:
        raise RuntimeError(
            "Stage2 checkpoint load has disallowed missing keys "
            f"(payload_stage={payload_stage!r}, bridge={bridge_from_stage1}): {bad_missing[:20]}"
        )
    if bad_unexpected:
        raise RuntimeError(
            "Stage2 checkpoint load has disallowed unexpected keys "
            f"(payload_stage={payload_stage!r}, bridge={bridge_from_stage1}): {bad_unexpected[:20]}"
        )


def stage2_action_mask_excludes_raw_images(
    action_mask: torch.Tensor,
    *,
    txt_len: int,
    synthetic_len: int,
    action_len: int,
) -> bool:
    """Stage2 action K/V layout is [txt | synthetic | action], with no raw image columns."""
    expected = (action_mask.shape[-2], action_mask.shape[-1])
    got = (action_len, txt_len + synthetic_len + action_len)
    if expected != got:
        raise ValueError(f"Stage2 action mask must be [..., {got[0]}, {got[1]}], got {tuple(action_mask.shape)}")
    return True
