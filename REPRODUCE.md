# Reproducing the LIT results on ImageWAM

This repository is the ImageWAM codebase plus a two-stage **Latent Interface Training (LIT)**
recipe on the FLUX.2 Klein 4B backbone. This page covers only what LIT adds; for the base
model see `README.md`.

## What LIT changes

The action expert is normally conditioned on backbone visual representations directly.
LIT replaces that path:

- **Stage 1** trains the action expert on language + robot state + each chunk's terminal
  SE(3) end-effector pose, with no image observations — a spatial-goal-conditioned action
  prior that never sees appearance.
- **Stage 2** restores vision, but only through 100 learnable latent tokens that aggregate
  the backbone; raw visual tokens are masked out of the action expert, and 8 of the latents
  are supervised to reconstruct the same terminal pose (`lambda_pose = 0.3`).

Stage 1's SE(3) encoder is training-time scaffolding — Stage 2 drops it and the latents
predict the pose from vision, so no privileged input is needed at inference.

## Baseline

The baseline is the released ImageWAM FLUX.2 Klein 4B checkpoint, evaluated as-is; we do not
retrain it. See `README.md` for the download.

## Training LIT

Configs: `configs/task/libero_flux2_klein_4b_goal_prior_stage{1,2}.yaml`, on all four LIBERO
suites (`libero_spatial / object / goal / 10`, `no_noops`).

```bash
# Stage 1 — vision-free SE(3)-conditioned action prior
#   batch 64, lr 2e-4, warmup 2000, 10000 steps
bash scripts/flux2/run_train_flux2_klein_goal_prior_stage1.sh

# Stage 2 — pose-supervised latent interface, initialised from Stage 1
#   batch 10, lr 1e-4, warmup = 5% of the schedule, 10 epochs
bash scripts/flux2/run_train_flux2_klein_goal_prior_stage2.sh
```

Interface settings are identical to the other three backbones and were not tuned per model:
`num_latents=100`, `num_pose_tokens=8`, `latent_dim=768`, `inner_dim=512`, `lambda_pose=0.3`.

## Evaluation

### LIBERO (in-distribution)

```bash
export CKPT_PATH=<stage2_run>/checkpoints/weights/step_034720.pt
export DATASET_STATS_PATH=<stage2_run>/dataset_stats.json
NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
```

### LIBERO-Plus (out of distribution)

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) is 10,030 perturbed tasks over seven
axes, one episode each.

```bash
export CKPT_PATH=<stage2_run>/checkpoints/weights/step_034720.pt
export DATASET_STATS_PATH=<stage2_run>/dataset_stats.json
LIBERO_PLUS_FIX_LANG=1 NUM_GPUS=8 FLUX2_VARIANT=4b \
  bash scripts/flux2/run_eval_flux2_libero_plus.sh
```

**`LIBERO_PLUS_FIX_LANG=1` is required.** Upstream LIBERO-Plus derives the instruction from
the perturbed file name, so on every non-language axis the policy is otherwise fed strings
like `... view 0 0 100 2 352 initstate 0`. Numbers produced without it are not comparable.

ImageWAM supports three conditioning modes at evaluation (`both` / `ref_only` / `syn_only`);
the reported numbers use `both`.

## Checkpoints

Released as `imagewam/lit_stage1` and `imagewam/lit_stage2` (Hugging Face, link in the LIT hub), each with
`model.pt`, `config.yaml` and `dataset_stats.json`: `CKPT_PATH=<dir>/model.pt`
`DATASET_STATS_PATH=<dir>/dataset_stats.json`. The baseline is the released ImageWAM FLUX.2 Klein 4B checkpoint.

## The same method on other backbones

π0.5 `jianmanlincjx/pi05` · MolmoAct2 `jianmanlincjx/Molmoact2` · FAST-WAM `jianmanlincjx/fastwam`
