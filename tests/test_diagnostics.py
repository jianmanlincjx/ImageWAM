"""Blackout at the shipped batch size, and the training diagnostics.

The property that matters operationally is that both regimes are present in
every batch on every rank: the trainer all-gathers once per metric key, so a
key that exists on some ranks and not others deadlocks the step.
"""
import torch

from imagewam.models.backbones.goal_pose_prior import SemanticVisualAggregator
from imagewam.models.backbones.imagewam import ImageWAM

DEV = torch.device("cpu")


def _agg(**kw):
    base = dict(num_tokens=100, latent_dim=64, context_dim=96, kv_dim=128,
                attn_head_dim=64, num_layer_groups=5, num_heads=4, num_pose_tokens=8)
    base.update(kw)
    return SemanticVisualAggregator(**base)


def test_shipped_config_always_yields_both_regimes():
    """batch 10 x p=0.10 must give exactly one dark sample, never zero."""
    a = _agg(context_blackout_prob=0.10, context_token_dropout=0.15)
    a.train()
    for _ in range(200):
        m = a.sample_context_keep_mask(10, DEV)
        dark = ~m[:, 8:].any(dim=1)
        assert int(dark.sum()) == 1, int(dark.sum())
        assert int((~dark).sum()) == 9
        assert m[:, :8].all()


def test_blackout_never_takes_the_whole_batch():
    a = _agg(context_blackout_prob=0.95)
    a.train()
    m = a.sample_context_keep_mask(4, DEV)
    dark = ~m[:, 8:].any(dim=1)
    assert int((~dark).sum()) >= 1, "at least one steered sample must survive"


def test_tiny_probability_still_blacks_out_one():
    a = _agg(context_blackout_prob=0.01)
    a.train()
    m = a.sample_context_keep_mask(10, DEV)
    assert int((~m[:, 8:].any(dim=1)).sum()) == 1


def test_blackout_selection_varies_between_steps():
    torch.manual_seed(0)
    a = _agg(context_blackout_prob=0.30)
    a.train()
    seen = set()
    for _ in range(50):
        m = a.sample_context_keep_mask(10, DEV)
        seen.add(tuple((~m[:, 8:].any(dim=1)).tolist()))
    assert len(seen) > 5, "which samples are dark must be random"


# --------------------------------------------------------------- diagnostics
class _Diag:
    """Only the attributes _goal_prior_diagnostics touches."""

    loss_lambda_action = 1.0
    _goal_prior_diagnostics = ImageWAM._goal_prior_diagnostics

    def __init__(self, agg):
        self.semantic_visual_aggregator = agg


def test_diagnostics_emit_every_key_even_with_no_dark_sample():
    """Key sets must be identical across ranks or the all-gather deadlocks."""
    a = _agg(context_blackout_prob=0.10)
    d = _Diag(a)
    keep_all_live = torch.ones(4, 100, dtype=torch.bool)
    loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = torch.ones(4)
    out = d._goal_prior_diagnostics(
        syn_keep_mask=keep_all_live, action_loss_per_sample=loss, action_weight=w
    )
    for k in ("blackout_frac", "loss_action_fallback", "loss_action_steered",
              "gate/bias_mean", "gate/bias_first", "gate/bias_last"):
        assert k in out, k
    assert out["blackout_frac"] == 0.0


def test_diagnostics_split_the_two_regimes():
    a = _agg(context_blackout_prob=0.25)
    d = _Diag(a)
    keep = torch.ones(4, 100, dtype=torch.bool)
    keep[1, 8:] = False          # sample 1 is the dark one
    loss = torch.tensor([1.0, 10.0, 3.0, 5.0])
    w = torch.ones(4)
    out = d._goal_prior_diagnostics(
        syn_keep_mask=keep, action_loss_per_sample=loss, action_weight=w
    )
    assert out["blackout_frac"] == 0.25
    assert out["loss_action_fallback"] == 10.0
    assert abs(out["loss_action_steered"] - 3.0) < 1e-6   # mean(1, 3, 5)


def test_diagnostics_report_the_gate_without_a_mask():
    a = _agg(gate_bias_init=-5.0)
    d = _Diag(a)
    out = d._goal_prior_diagnostics(
        syn_keep_mask=None,
        action_loss_per_sample=torch.zeros(2),
        action_weight=torch.ones(2),
    )
    assert out["gate/bias_mean"] == -5.0
    assert "loss_action_fallback" not in out, "no mask, no regime split"
