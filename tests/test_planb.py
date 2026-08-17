"""Plan B: the reference channel, the three regimes, and the mask layout.

The layout is [txt | ref | syn | action]. The single most dangerous mistake is
the gate's column offset: it used to be `txt_len` because the synthetic block
started right after the text, and with the reference block in between a stale
offset silently biases the *reference* columns instead -- suppressing the clean
channel and letting the gated one through, i.e. exactly backwards.
"""
import torch

from imagewam.models.backbones.goal_pose_prior import (
    SemanticVisualAggregator,
    build_stage2_action_attention_mask,
)
from imagewam.models.backbones.mot import MoT

DEV = torch.device("cpu")


def _agg(**kw):
    base = dict(num_tokens=100, latent_dim=64, context_dim=96, kv_dim=128,
                attn_head_dim=64, num_layer_groups=5, num_heads=4, num_pose_tokens=8)
    base.update(kw)
    return SemanticVisualAggregator(**base)


# ------------------------------------------------------------ mask layout
def test_mask_layout_is_txt_ref_syn_action():
    B, T, R, S, A = 1, 4, 6, 10, 3
    m = build_stage2_action_attention_mask(B, T, S, A, DEV, ref_len=R)
    assert m.shape == (B, A, T + R + S + A)
    assert m.all(), "everything visible when nothing is dropped"


def test_ref_len_zero_reproduces_the_firewalled_mask():
    B, T, S, A = 2, 5, 8, 3
    a = build_stage2_action_attention_mask(B, T, S, A, DEV)
    b = build_stage2_action_attention_mask(B, T, S, A, DEV, ref_len=0)
    assert torch.equal(a, b)


def test_ref_keep_mask_drops_only_the_ref_block():
    B, T, R, S, A = 3, 4, 5, 6, 2
    ref_keep = torch.tensor([True, False, True])
    m = build_stage2_action_attention_mask(B, T, S, A, DEV, ref_len=R, ref_keep_mask=ref_keep)
    assert not m[1, :, T : T + R].any(), "ref-dropped sample sees no reference"
    assert m[1, :, :T].all() and m[1, :, T + R :].all(), "other blocks untouched"
    assert m[0, :, T : T + R].all() and m[2, :, T : T + R].all()


def test_syn_keep_mask_lands_after_the_ref_block():
    B, T, R, S, A = 1, 3, 7, 5, 2
    keep = torch.ones(B, S, dtype=torch.bool)
    keep[0, 2] = False
    m = build_stage2_action_attention_mask(B, T, S, A, DEV, synthetic_keep_mask=keep, ref_len=R)
    assert not m[0, :, T + R + 2].any(), "dropped synthetic column"
    assert m[0, :, T + 2].all(), "the ref block must not be hit by the syn mask"


def test_ref_keep_without_ref_len_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="ref_keep_mask"):
        build_stage2_action_attention_mask(
            2, 3, 4, 2, DEV, ref_keep_mask=torch.ones(2, dtype=torch.bool)
        )


# --------------------------------------------------------- the gate offset
def test_gate_offset_skips_the_ref_block():
    """The regression that would suppress the clean channel instead."""
    T, R, S, A, POSE = 4, 20, 100, 2, 8
    bool_mask = torch.ones(1, A, T + R + S + A, dtype=torch.bool)
    m = MoT._flux2_gate_action_mask(bool_mask, T + R, S, torch.tensor(-2.0), (POSE, S))
    assert torch.all(m[:, :, T : T + R] == 0.0), "reference columns must stay unbiased"
    assert torch.all(m[:, :, T + R : T + R + POSE] == 0.0), "pose columns unbiased"
    assert torch.all(m[:, :, T + R + POSE : T + R + S] == -2.0), "context columns biased"
    assert torch.all(m[:, :, :T] == 0.0)


def test_stale_offset_would_have_been_caught():
    """Sanity: passing the old txt_len offset does hit the reference block."""
    T, R, S, A, POSE = 4, 20, 100, 2, 8
    bool_mask = torch.ones(1, A, T + R + S + A, dtype=torch.bool)
    wrong = MoT._flux2_gate_action_mask(bool_mask, T, S, torch.tensor(-2.0), (POSE, S))
    assert (wrong[:, :, T : T + R] == -2.0).any(), "this is the bug the test above guards"


# --------------------------------------------------------- regime sampling
def test_regimes_are_all_present_at_the_shipped_batch():
    a = _agg(action_sees_ref=True, p_both=0.55, p_ref_only=0.15, p_syn_only=0.30)
    a.train()
    for _ in range(200):
        ref_keep, syn_keep = a.sample_channel_regime(10, DEV)
        both = ref_keep & syn_keep
        ref_only = ref_keep & ~syn_keep
        syn_only = ~ref_keep & syn_keep
        assert int(both.sum()) >= 1
        assert int(ref_only.sum()) >= 1
        assert int(syn_only.sum()) >= 1
        assert int((~ref_keep & ~syn_keep).sum()) == 0, "never blind both channels"
        assert int(both.sum() + ref_only.sum() + syn_only.sum()) == 10


def test_regime_proportions_are_close_to_config():
    torch.manual_seed(0)
    a = _agg(action_sees_ref=True, p_both=0.55, p_ref_only=0.15, p_syn_only=0.30)
    a.train()
    n = 0
    tally = {"both": 0, "ref_only": 0, "syn_only": 0}
    for _ in range(200):
        ref_keep, syn_keep = a.sample_channel_regime(64, DEV)
        tally["both"] += int((ref_keep & syn_keep).sum())
        tally["ref_only"] += int((ref_keep & ~syn_keep).sum())
        tally["syn_only"] += int((~ref_keep & syn_keep).sum())
        n += 64
    assert abs(tally["both"] / n - 0.55) < 0.03, tally["both"] / n
    assert abs(tally["ref_only"] / n - 0.15) < 0.03, tally["ref_only"] / n
    assert abs(tally["syn_only"] / n - 0.30) < 0.03, tally["syn_only"] / n


def test_regime_sampling_is_off_when_disabled_or_evaluating():
    a = _agg(action_sees_ref=False)
    a.train()
    assert a.sample_channel_regime(8, DEV) == (None, None)
    b = _agg(action_sees_ref=True)
    b.eval()
    assert b.sample_channel_regime(8, DEV) == (None, None)


def test_regime_probabilities_must_sum_to_one():
    import pytest

    with pytest.raises(ValueError, match="sum to 1"):
        _agg(action_sees_ref=True, p_both=0.5, p_ref_only=0.2, p_syn_only=0.4)


# ------------------------------------------- the two masks compose correctly
def test_ref_only_sample_loses_every_synthetic_column():
    """Channel regime and context blackout must intersect, not overwrite."""
    a = _agg(action_sees_ref=True, context_blackout_prob=0.10, context_token_dropout=0.05)
    a.train()
    ref_keep, syn_keep = a.sample_channel_regime(10, DEV)
    ctx = a.sample_context_keep_mask(10, DEV)
    combined = ctx & syn_keep[:, None]
    for i in range(10):
        if not bool(syn_keep[i]):
            assert not combined[i].any(), "ref-only sample must see no synthetic token"
        else:
            assert combined[i, :8].all(), "pose survives unless the channel is off"
