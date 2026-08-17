"""Tests for the bottleneck fixes (B1 grid latents, B2 context dropout, B4 gate).

The governing invariant: with every flag at its default the module must behave
exactly as the evaluated revision did, so this branch can be trained without
silently changing the control.
"""
import torch

from imagewam.models.backbones.goal_pose_prior import (
    GOAL_PRIOR_CONTEXT_TOKEN_DROPOUT,
    GOAL_PRIOR_LATENT_LAYOUT,
    GOAL_PRIOR_SYN_GATE_BIAS_INIT,
    GOAL_PRIOR_ZERO_INIT_VALUE,
    SemanticVisualAggregator,
    build_stage2_action_attention_mask,
)

DEV = torch.device("cpu")


def _agg(**kw):
    base = dict(num_tokens=100, latent_dim=64, context_dim=96, kv_dim=128,
                attn_head_dim=64, num_layer_groups=5, num_heads=4)
    base.update(kw)
    return SemanticVisualAggregator(**base)


# --------------------------------------------------------------- defaults off
def test_defaults_match_evaluated_revision():
    assert GOAL_PRIOR_CONTEXT_TOKEN_DROPOUT == 0.0
    assert GOAL_PRIOR_ZERO_INIT_VALUE is False
    assert GOAL_PRIOR_SYN_GATE_BIAS_INIT == 0.0
    assert GOAL_PRIOR_LATENT_LAYOUT == "free"
    a = _agg()
    assert a.sample_context_keep_mask(4, DEV, training=True) is None
    assert a.visual_attn_mask(784, DEV) is None
    assert not torch.allclose(a.groups[0].to_value.weight, torch.zeros(1))


# ------------------------------------------------------------ B2: ctx dropout
def test_context_dropout_never_touches_pose_tokens():
    a = _agg(context_token_dropout=0.9, num_pose_tokens=8)
    m = a.sample_context_keep_mask(64, DEV, training=True)
    assert m.shape == (64, 100) and m.dtype == torch.bool
    assert m[:, :8].all(), "pose tokens must never be dropped"
    dropped = (~m[:, 8:]).float().mean().item()
    assert 0.8 < dropped < 1.0, dropped


def test_context_dropout_off_at_eval():
    a = _agg(context_token_dropout=0.5)
    assert a.sample_context_keep_mask(4, DEV, training=False) is None


def test_action_mask_applies_syn_dropout_only_to_syn_columns():
    B, T, S, A = 2, 5, 10, 3
    keep = torch.ones(B, S, dtype=torch.bool)
    keep[0, 3] = False
    keep[1, 7] = False
    m = build_stage2_action_attention_mask(B, T, S, A, DEV, synthetic_keep_mask=keep)
    assert m.shape == (B, A, T + S + A)
    assert m[:, :, :T].all(), "text columns untouched"
    assert m[:, :, T + S :].all(), "action self-attention untouched"
    assert not m[0, :, T + 3].any() and m[0, :, T + 4].all()
    assert not m[1, :, T + 7].any() and m[1, :, T + 3].all()


def test_action_mask_unchanged_without_keep_mask():
    B, T, S, A = 2, 4, 6, 3
    a = build_stage2_action_attention_mask(B, T, S, A, DEV)
    b = build_stage2_action_attention_mask(B, T, S, A, DEV, synthetic_keep_mask=None)
    assert torch.equal(a, b) and a.all()


# ------------------------------------------------------------------ B4: gate
def test_zero_init_value_and_gate_bias():
    a = _agg(zero_init_value=True, gate_bias_init=-5.0)
    for g in a.groups:
        assert torch.count_nonzero(g.to_value.weight) == 0
        assert float(g.syn_gate_bias) == -5.0
    # to_key must stay live: a zeroed key would make the columns identical
    assert torch.count_nonzero(a.groups[0].to_key.weight) > 0
    assert float(a.gate_bias(0, 25)) == -5.0
    assert a.gate_bias(24, 25) is a.groups[4].syn_gate_bias


def test_gate_bias_is_learnable():
    a = _agg(gate_bias_init=-5.0)
    assert a.groups[0].syn_gate_bias.requires_grad


# ------------------------------------------------------------------ B1: grid
def test_grid_mask_geometry():
    a = _agg(latent_layout="grid", grid_hw=(7, 7), grid_window=0, num_pose_tokens=8)
    m = a.visual_attn_mask(784, DEV)          # 28x28 image tokens
    assert m.shape == (100, 784) and m.dtype == torch.bool
    assert not m[:8].any(), "pose tokens keep global access"
    assert not m[8 + 49 :].any(), "spare latents keep global access"
    for cell in range(49):
        row = m[8 + cell]
        visible = int((~row).sum())
        assert visible == 16, f"cell {cell} sees {visible} tokens, expected 4x4=16"
    # every image token is covered by exactly one cell at window=0
    cover = (~m[8 : 8 + 49]).sum(dim=0)
    assert int(cover.min()) == 1 and int(cover.max()) == 1


def test_grid_window_widens_neighbourhood():
    a0 = _agg(latent_layout="grid", grid_hw=(7, 7), grid_window=0)
    a1 = _agg(latent_layout="grid", grid_hw=(7, 7), grid_window=1)
    n0 = int((~a0.visual_attn_mask(784, DEV)[8]).sum())
    n1 = int((~a1.visual_attn_mask(784, DEV)[8]).sum())
    assert n1 > n0, (n0, n1)


def test_grid_falls_back_when_token_count_is_not_square():
    a = _agg(latent_layout="grid")
    assert a.visual_attn_mask(783, DEV) is None, "must not guess a layout"


def test_grid_mask_is_cached():
    a = _agg(latent_layout="grid")
    assert a.visual_attn_mask(784, DEV) is a.visual_attn_mask(784, DEV)


def test_grid_rejects_impossible_config():
    import pytest

    with pytest.raises(ValueError, match="grid layout needs"):
        _agg(latent_layout="grid", grid_hw=(16, 16), num_tokens=100, num_pose_tokens=8)


def test_invalid_layout_rejected():
    import pytest

    with pytest.raises(ValueError, match="latent_layout"):
        _agg(latent_layout="spatial")


# ------------------------------------------------- grid actually restricts flow
def test_grid_blocks_distant_image_tokens_end_to_end():
    """A change confined to one corner must not move a latent bound elsewhere."""
    torch.manual_seed(0)
    a = _agg(latent_layout="grid", grid_hw=(7, 7), grid_window=0, num_pose_tokens=8)
    a.eval()
    B, ctx = 1, 784
    txt = torch.randn(B, 6, 96)
    img = torch.randn(B, ctx, 96)
    tmask = torch.ones(B, 6, dtype=torch.bool)
    imask = torch.ones(B, ctx, dtype=torch.bool)
    amask = a.visual_attn_mask(ctx, DEV)
    q = a.initial_queries(B)

    def run(image):
        with torch.no_grad():
            return a.forward_layer(q, txt, image, semantic_mask=tmask, image_mask=imask,
                                   layer_idx=0, num_layers=25, visual_attn_mask=amask)

    base = run(img)
    img2 = img.clone()
    # Resample the token rather than shifting it: the block LayerNorms the
    # context, so a constant added across the feature dim would be normalised
    # straight back out and the test would pass for the wrong reason.
    img2[:, 0] = torch.randn_like(img2[:, 0]) * 5.0
    moved = (run(img2) - base).abs().amax(dim=-1)[0]

    far = 8 + 48                # bottom-right cell
    assert moved[far] < 1e-5, f"far latent moved by {moved[far]:.3e}"
    near = 8 + 0                # top-left cell owns token 0
    assert moved[near] > 1e-4, f"near latent did not move ({moved[near]:.3e})"


def test_context_dropout_follows_module_mode_by_default():
    """Regression: a default of training=True would drop tokens at inference."""
    a = _agg(context_token_dropout=0.5)
    a.train()
    assert a.sample_context_keep_mask(4, DEV) is not None
    a.eval()
    assert a.sample_context_keep_mask(4, DEV) is None, "must not drop at inference"


# ------------------------------------------------------- B2+: context blackout
def test_blackout_masks_whole_context_but_keeps_pose():
    torch.manual_seed(0)
    a = _agg(context_token_dropout=0.0, context_blackout_prob=0.5, num_pose_tokens=8)
    a.train()
    m = a.sample_context_keep_mask(512, DEV)
    assert m[:, :8].all(), "pose tokens survive blackout"
    ctx_alive = m[:, 8:].any(dim=1)
    frac = 1.0 - float(ctx_alive.float().mean())
    assert 0.4 < frac < 0.6, frac
    # a blacked-out sample loses the entire context block, not part of it
    for i in range(512):
        if not bool(ctx_alive[i]):
            assert not m[i, 8:].any()


def test_blackout_composes_with_per_token_dropout():
    torch.manual_seed(0)
    a = _agg(context_token_dropout=0.2, context_blackout_prob=0.25, num_pose_tokens=8)
    a.train()
    m = a.sample_context_keep_mask(1024, DEV)
    assert m[:, :8].all()
    fully_dark = ~m[:, 8:].any(dim=1)
    partial = m[:, 8:].any(dim=1) & ~m[:, 8:].all(dim=1)
    assert 0.15 < float(fully_dark.float().mean()) < 0.35
    assert float(partial.float().mean()) > 0.5, "per-token dropout still active"


def test_blackout_alone_enables_the_mask():
    a = _agg(context_token_dropout=0.0, context_blackout_prob=0.1)
    a.train()
    assert a.sample_context_keep_mask(8, DEV) is not None
    a.eval()
    assert a.sample_context_keep_mask(8, DEV) is None


def test_blackout_rejects_bad_prob():
    import pytest

    with pytest.raises(ValueError, match="context_blackout_prob"):
        _agg(context_blackout_prob=1.0)
