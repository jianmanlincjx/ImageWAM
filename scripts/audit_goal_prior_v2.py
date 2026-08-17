#!/usr/bin/env python3
"""End-to-end audit: does the shipped config actually produce the intended model?

Checks the wiring, not the unit behaviour -- the tests cover the latter. Every
assertion here is something that would silently train the wrong thing.
"""
import sys
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/data2/JM/Code/ImageWAM-v2/src")
from imagewam.models.backbones.goal_pose_prior import (  # noqa: E402
    SemanticVisualAggregator, build_stage2_action_attention_mask,
    STAGE1_ONLY_CHECKPOINT_PREFIXES,
)
from imagewam.models.backbones.mot import MoT  # noqa: E402

ROOT = "/data2/JM/Code/ImageWAM-v2/configs"
ok, bad = [], []


def check(cond, msg):
    (ok if cond else bad).append(msg)


# ---------------------------------------------------------------- 1. configs
s2 = OmegaConf.load(f"{ROOT}/model/imagewam_flux2_klein_4b_goal_prior_stage2.yaml")
s1 = OmegaConf.load(f"{ROOT}/model/imagewam_flux2_klein_4b_goal_prior_stage1.yaml")
t2 = OmegaConf.load(f"{ROOT}/task/libero_flux2_klein_4b_goal_prior_stage2.yaml")
t1 = OmegaConf.load(f"{ROOT}/task/libero_flux2_klein_4b_goal_prior_stage1.yaml")
gp2, gp1 = s2.goal_prior, s1.goal_prior

check(float(gp2.context_token_dropout) == 0.05, "B2  per-token dropout = 0.05")
check(float(gp2.context_blackout_prob) == 0.10, "B2+ context blackout = 0.10")
check(bool(gp2.zero_init_value) is False, "B4  zero_init_value off (would silence pose)")
check(float(gp2.syn_gate_bias_init) == -2.0, "B4  gate bias = -2.0")
check(bool(gp2.gate_pose_tokens) is False, "B4  gate covers context only")
check(str(gp2.latent_layout) == "free", "B1  grid layout NOT enabled this round")
check(bool(gp1.stage1_sample_video_timestep) is True, "A1  stage1 samples video timestep")
check(int(gp1.stage1_null_image_tokens) == 32, "A2  32 null-image tokens")
check(t2.warmup_steps is None, "A4  stage2 warmup null -> 1736, matches baseline")
check(int(t1.batch_size) == 64, "S1-1 stage1 batch = 64")
check(int(t1.max_steps) == 10000, "S1-1 stage1 steps still 10k (loss had not converged)")
check(float(t1.learning_rate) == 2e-4, "S1-1 stage1 LR unchanged at 2e-4")

# ------------------------------------------------- 2. the model that results
agg = SemanticVisualAggregator(
    num_tokens=int(gp2.num_latents), latent_dim=int(gp2.latent_dim),
    context_dim=3072, kv_dim=3072, attn_head_dim=128,
    num_layer_groups=int(gp2.num_layer_groups),
    num_pose_tokens=int(gp2.num_pose_tokens),
    context_token_dropout=float(gp2.context_token_dropout),
    context_blackout_prob=float(gp2.context_blackout_prob),
    zero_init_value=bool(gp2.zero_init_value),
    gate_bias_init=float(gp2.syn_gate_bias_init),
    gate_pose_tokens=bool(gp2.gate_pose_tokens),
    latent_layout=str(gp2.latent_layout),
    grid_hw=tuple(gp2.grid_hw), grid_window=int(gp2.grid_window),
    action_sees_ref=bool(gp2.action_sees_ref),
    p_both=float(gp2.p_both), p_ref_only=float(gp2.p_ref_only),
    p_syn_only=float(gp2.p_syn_only),
)
check(agg.gated_span() == (8, 100), "gate span = (8,100): pose open, context gated")
check(torch.count_nonzero(agg.groups[0].to_value.weight) > 0, "pose value path is live")
check(all(float(g.syn_gate_bias) == -2.0 for g in agg.groups), "all 5 groups carry the gate")
check(all(g.syn_gate_bias.requires_grad for g in agg.groups), "gate is learnable (can open)")
check(agg.visual_attn_mask(784, torch.device("cpu")) is None, "free layout -> no grid mask")

# --------------------------------------------- 3. B2/B2+ actually mask things
agg.train()
m = agg.sample_context_keep_mask(4096, torch.device("cpu"))
check(m is not None and bool(m[:, :8].all()), "pose columns never dropped")
blacked = ~m[:, 8:].any(dim=1)
check(0.07 < float(blacked.float().mean()) < 0.13, "blackout rate ~10%")
partial = m[:, 8:].any(dim=1) & ~m[:, 8:].all(dim=1)
check(float(partial.float().mean()) > 0.5, "per-token dropout still active on the rest")
agg.eval()
check(agg.sample_context_keep_mask(4, torch.device("cpu")) is None, "no dropout at inference")

# ------------------------------------- 4. the two mechanisms agree on fallback
# B2+ fallback state and B4 step-0 state must be the same set of live columns.
T, A = 513, 16
keep = torch.ones(1, 100, dtype=torch.bool)
keep[0, 8:] = False                                   # a blacked-out sample
mask = build_stage2_action_attention_mask(1, T, 100, A, torch.device("cpu"),
                                          synthetic_keep_mask=keep)
live_blackout = {i for i in range(100) if bool(mask[0, 0, T + i])}
lo, hi = agg.gated_span()
live_gate = set(range(0, lo)) | set(range(hi, 100))   # ungated columns at step 0
check(live_blackout == live_gate == set(range(8)),
      "B2+ fallback and B4 step-0 both leave exactly the 8 pose columns live")

# ------------------------------------------------- 5. gate mask numerics
bm = torch.ones(1, A, T + 100 + A, dtype=torch.bool)
gm = MoT._flux2_gate_action_mask(bm, T, 100, torch.tensor(-5.0), agg.gated_span())
check(gm.dtype == torch.float32, "gate produces a float additive mask")
check(bool(torch.all(gm[:, :, T : T + 8] == 0.0)), "pose columns unbiased")
check(bool(torch.all(gm[:, :, T + 8 : T + 100] == -5.0)), "context columns biased -5")
check(bool(torch.all(gm[:, :, :T] == 0.0)), "text columns untouched")
check(bool(torch.all(gm[:, :, T + 100 :] == 0.0)), "action self-attention untouched")
fm = MoT._format_attention_mask(gm, 1, A, T + 100 + A, torch.device("cpu"))
check(fm.dtype == torch.float32, "float mask survives the formatter (gate would be lost)")

# ------------------------------------------------------ 6. checkpoint hygiene
check(any("stage1_null_image_tokens" in p for p in STAGE1_ONLY_CHECKPOINT_PREFIXES),
      "A2 parameter is declared stage1-only for the fail-closed bridge")

# --------------------------------------------------- 7. plan B: the ref channel
check(bool(gp2.action_sees_ref) is True, "B  Action Expert attends the reference image")
check(abs(float(gp2.p_both) + float(gp2.p_ref_only) + float(gp2.p_syn_only) - 1.0) < 1e-9,
      "B  regime probabilities sum to 1")
check(float(gp2.p_syn_only) >= 0.25,
      "B  syn-only regime trained enough to report a firewall number")
check(float(gp2.syn_gate_bias_init) == -2.0,
      "B  gate bias relaxed to -2 (784 ref keys already dilute the syn columns)")
check(float(gp2.context_token_dropout) == 0.05,
      "B  per-token dropout reduced; channel dropout now carries the pressure")

agg.train()
rk, sk = agg.sample_channel_regime(10, torch.device("cpu"))
check(rk is not None and int((rk & sk).sum()) >= 1 and int((rk & ~sk).sum()) >= 1
      and int((~rk & sk).sum()) >= 1,
      "B  every regime present in a batch of 10 (per-key all-gather safety)")
check(int((~rk & ~sk).sum()) == 0, "B  never blind both channels at once")
agg.eval()
check(agg.sample_channel_regime(10, torch.device("cpu")) == (None, None),
      "B  no channel dropout at inference")

T, R, S, A, POSE = 513, 784, 100, 16, 8
bm = torch.ones(1, A, T + R + S + A, dtype=torch.bool)
gm = MoT._flux2_gate_action_mask(bm, T + R, S, torch.tensor(-2.0), agg.gated_span())
check(bool(torch.all(gm[:, :, T : T + R] == 0.0)),
      "B  gate offset skips the ref block (a stale offset would suppress the clean channel)")
check(bool(torch.all(gm[:, :, T + R : T + R + POSE] == 0.0)), "B  pose columns still open")
check(bool(torch.all(gm[:, :, T + R + POSE : T + R + S] == -2.0)), "B  context columns gated")

m = build_stage2_action_attention_mask(2, T, S, A, torch.device("cpu"), ref_len=R,
                                       ref_keep_mask=torch.tensor([True, False]))
check(not bool(m[1, :, T : T + R].any()) and bool(m[1, :, :T].all()),
      "B  ref dropout hits only the ref block")

print("\n".join(f"  PASS  {m}" for m in ok))
if bad:
    print("\n".join(f"  FAIL  {m}" for m in bad))
    sys.exit(f"\n{len(bad)} FAILED / {len(ok)+len(bad)}")
print(f"\nall {len(ok)} wiring checks passed")
