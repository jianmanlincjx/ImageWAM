"""Regression tests for float (additive) masks flowing through _mixed_attention.

Both bugs these cover only surface at run time: SDPA rejects a float mask whose
dtype differs from the query, and the attention-capture path used `~mask`,
which is a bitwise not and throws on floats. Mask-level tests cannot catch
either, so these push real tensors through the attention.
"""
import torch

from imagewam.models.backbones.mot import MoT


class _Attn(MoT):
    """Minimal stand-in: _mixed_attention only needs the head geometry."""

    def __init__(self, num_heads=4, head_dim=8):
        torch.nn.Module.__init__(self)
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.attn_head_dim = head_dim
        self.gqa_implementation = "repeat"
        self.force_flash_attention = False
        self.mot_checkpoint_mixed_attn = False


def _qkv(B, Q, K, H, D, dtype):
    return (
        torch.randn(B, Q, H * D, dtype=dtype),
        torch.randn(B, K, H * D, dtype=dtype),
        torch.randn(B, K, H * D, dtype=dtype),
    )


def test_float_mask_runs_in_bfloat16():
    """SDPA needs the additive mask in the query dtype."""
    torch.manual_seed(0)
    m = _Attn()
    B, Q, T, S, A = 1, 4, 6, 10, 4
    K = T + S + A
    q, k, v = _qkv(B, Q, K, m.num_heads, m.attn_head_dim, torch.bfloat16)
    bool_mask = torch.ones(B, Q, K, dtype=torch.bool)
    gated = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(-5.0), (2, S))
    assert gated.dtype == torch.float32
    out = m._mixed_attention(q, k, v, gated, checkpoint=False)
    assert out.shape == (B, Q, m.num_heads * m.attn_head_dim)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_float_mask_matches_bool_mask_when_bias_is_zero():
    torch.manual_seed(0)
    m = _Attn()
    B, Q, T, S, A = 2, 3, 5, 8, 3
    K = T + S + A
    q, k, v = _qkv(B, Q, K, m.num_heads, m.attn_head_dim, torch.float32)
    bool_mask = torch.ones(B, Q, K, dtype=torch.bool)
    bool_mask[0, :, 2] = False
    zero_bias = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(0.0))
    a = m._mixed_attention(q, k, v, bool_mask, checkpoint=False)
    b = m._mixed_attention(q, k, v, zero_bias, checkpoint=False)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_gate_actually_suppresses_the_context_columns():
    """With a strong negative bias the output must approach the ungated-only case."""
    torch.manual_seed(0)
    m = _Attn()
    B, Q, T, S, A, POSE = 1, 2, 4, 20, 2, 3
    K = T + S + A
    q, k, v = _qkv(B, Q, K, m.num_heads, m.attn_head_dim, torch.float32)
    full = torch.ones(B, Q, K, dtype=torch.bool)

    strong = MoT._flux2_gate_action_mask(full, T, S, torch.tensor(-30.0), (POSE, S))
    gated_out = m._mixed_attention(q, k, v, strong, checkpoint=False)

    # ground truth: context columns hard-masked
    hard = full.clone()
    hard[:, :, T + POSE : T + S] = False
    hard_out = m._mixed_attention(q, k, v, hard, checkpoint=False)

    assert torch.allclose(gated_out, hard_out, atol=1e-4), (gated_out - hard_out).abs().max()


def test_attention_capture_accepts_a_float_mask():
    """`~mask` would throw here; the capture path is what the probes use."""
    torch.manual_seed(0)
    m = _Attn()
    B, Q, T, S, A, POSE = 1, 2, 3, 12, 2, 4
    K = T + S + A
    q, k, v = _qkv(B, Q, K, m.num_heads, m.attn_head_dim, torch.float32)
    full = torch.ones(B, Q, K, dtype=torch.bool)
    gated = MoT._flux2_gate_action_mask(full, T, S, torch.tensor(-5.0), (POSE, S))

    out, probs = m._mixed_attention(q, k, v, gated, return_attn_probs=True, checkpoint=False)
    assert out.shape == (B, Q, m.num_heads * m.attn_head_dim)
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-4)
    # pose columns must keep far more mass than the suppressed context ones
    pose_mass = probs[..., T : T + POSE].sum(-1).mean()
    ctx_mass = probs[..., T + POSE : T + S].sum(-1).mean()
    assert float(pose_mass) > 10 * float(ctx_mass), (float(pose_mass), float(ctx_mass))


def test_bool_mask_path_is_unchanged():
    torch.manual_seed(0)
    m = _Attn()
    B, Q, K = 2, 3, 9
    q, k, v = _qkv(B, Q, K, m.num_heads, m.attn_head_dim, torch.float32)
    mask = torch.ones(B, Q, K, dtype=torch.bool)
    mask[1, :, 4] = False
    out, probs = m._mixed_attention(q, k, v, mask, return_attn_probs=True, checkpoint=False)
    assert float(probs[1, 0, 0, 4]) == 0.0
    assert torch.isfinite(out).all()
