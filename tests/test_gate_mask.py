"""Tests for the MoT-level pieces of the B4 gate.

These cover the two places the gate can silently break: the shared mask
formatter (which used to hard-cast everything to bool) and the additive bias
itself, which must block what the boolean mask blocked and bias only the
synthetic columns.
"""
import math

import torch

from imagewam.models.backbones.mot import MoT


def test_gate_mask_biases_only_synthetic_columns():
    B, A, T, S, AL = 2, 3, 4, 5, 3
    bool_mask = torch.ones(B, A, T + S + AL, dtype=torch.bool)
    bool_mask[0, :, 1] = False           # a blocked text column
    bool_mask[1, :, T + 2] = False       # a dropped synthetic column (B2)
    bias = torch.tensor(-5.0)

    m = MoT._flux2_gate_action_mask(bool_mask, T, S, bias)

    assert m.dtype == torch.float32
    # blocked stays blocked, even inside the biased span
    assert math.isinf(m[0, 0, 1]) and m[0, 0, 1] < 0
    assert math.isinf(m[1, 0, T + 2]) and m[1, 0, T + 2] < 0
    # synthetic columns are biased down, text and action are untouched
    assert torch.allclose(m[0, :, T : T + S][m[0, :, T : T + S] > -1e30], torch.tensor(-5.0))
    assert torch.all(m[0, :, :T][bool_mask[0, :, :T]] == 0.0)
    assert torch.all(m[:, :, T + S :] == 0.0)


def test_gate_mask_zero_bias_is_a_plain_additive_mask():
    B, A, T, S, AL = 1, 2, 3, 4, 2
    bool_mask = torch.ones(B, A, T + S + AL, dtype=torch.bool)
    bool_mask[0, :, 0] = False
    m = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(0.0))
    allowed = m > -1e30
    assert torch.all(m[allowed] == 0.0)
    assert torch.equal(allowed, bool_mask)


def test_format_attention_mask_preserves_float_and_bool():
    B, Q, K = 2, 3, 7
    dev = torch.device("cpu")

    f = torch.zeros(B, Q, K)
    f[:, :, 0] = float("-inf")
    out = MoT._format_attention_mask(f, B, Q, K, dev)
    assert out.dtype == torch.float32 and out.shape == (B, 1, Q, K)
    assert math.isinf(out[0, 0, 0, 0])

    b = torch.ones(B, Q, K, dtype=torch.bool)
    out_b = MoT._format_attention_mask(b, B, Q, K, dev)
    assert out_b.dtype == torch.bool and out_b.shape == (B, 1, Q, K)


def test_format_attention_mask_still_validates_shape():
    import pytest

    with pytest.raises(ValueError, match="3D attention mask"):
        MoT._format_attention_mask(
            torch.zeros(2, 3, 9), 2, 3, 7, torch.device("cpu")
        )


def test_gate_mask_matches_boolean_semantics_under_softmax():
    """A -inf entry must contribute exactly zero probability."""
    torch.manual_seed(0)
    T, S, AL = 3, 4, 2
    bool_mask = torch.ones(1, 1, T + S + AL, dtype=torch.bool)
    bool_mask[0, 0, 2] = False
    logits = torch.randn(1, 1, T + S + AL)

    m = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(-5.0))
    p = torch.softmax(logits + m, dim=-1)
    assert float(p[0, 0, 2]) == 0.0
    # the biased span must lose mass relative to no bias
    p0 = torch.softmax(logits + MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(0.0)), dim=-1)
    assert p[0, 0, T : T + S].sum() < p0[0, 0, T : T + S].sum()


def test_gate_span_biases_only_the_context_columns():
    """Split gate: pose columns keep a zero bias, context columns are suppressed."""
    B, A, T, S, AL, POSE = 1, 2, 4, 100, 3, 8
    bool_mask = torch.ones(B, A, T + S + AL, dtype=torch.bool)
    m = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(-5.0), gated_span=(POSE, S))
    assert torch.all(m[:, :, T : T + POSE] == 0.0), "pose columns unbiased"
    assert torch.all(m[:, :, T + POSE : T + S] == -5.0), "context columns biased"
    assert torch.all(m[:, :, :T] == 0.0)
    assert torch.all(m[:, :, T + S :] == 0.0)


def test_gate_span_defaults_to_the_whole_block():
    B, A, T, S, AL = 1, 1, 2, 6, 2
    bool_mask = torch.ones(B, A, T + S + AL, dtype=torch.bool)
    m = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(-3.0))
    assert torch.all(m[:, :, T : T + S] == -3.0)


def test_split_gate_keeps_pose_share_under_softmax():
    """Pose must retain real probability mass while context is suppressed."""
    T, S, AL, POSE = 4, 100, 3, 8
    bool_mask = torch.ones(1, 1, T + S + AL, dtype=torch.bool)
    logits = torch.zeros(1, 1, T + S + AL)
    m = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(-5.0), gated_span=(POSE, S))
    p = torch.softmax(logits + m, dim=-1)
    pose_mass = float(p[0, 0, T : T + POSE].sum())
    ctx_mass = float(p[0, 0, T + POSE : T + S].sum())
    assert pose_mass > 10 * ctx_mass, (pose_mass, ctx_mass)


def test_blocked_columns_stay_blocked_inside_the_gated_span():
    """B2 dropout and B4 gate compose: a dropped column is -inf, not merely biased."""
    T, S, AL, POSE = 3, 100, 2, 8
    bool_mask = torch.ones(1, 1, T + S + AL, dtype=torch.bool)
    bool_mask[0, :, T + 40] = False          # a context column dropped by B2
    m = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(-5.0), gated_span=(POSE, S))
    assert m[0, 0, T + 40] == float("-inf")
    assert m[0, 0, T + 41] == -5.0
