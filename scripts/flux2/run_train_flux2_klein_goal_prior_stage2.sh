#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."
# Keep goal-prior runs offline even if the shell/default train.yaml wants online.
export WANDB_MODE=offline

GPU_PER_NODE="${GPU_PER_NODE:-8}"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"
ZERO_STAGE="${ZERO_STAGE:-1}"
PRECOMPUTE_QWEN3_CACHE="${PRECOMPUTE_QWEN3_CACHE:-false}"
QWEN_CONTEXT_LEN="${QWEN_CONTEXT_LEN:-512}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"

imagewam_require_env DATA_ROOT
imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env STAGE1_CHECKPOINT

if [ "${FLUX2_VARIANT}" != "4b" ]; then
  echo "Goal-pose prior Stage2 currently supports FLUX2_VARIANT=4b only, got ${FLUX2_VARIANT}" >&2
  exit 1
fi

MODEL_CONFIG="configs/model/imagewam_flux2_klein_4b_goal_prior_stage2.yaml"
TASK_NAME="libero_flux2_klein_4b_goal_prior_stage2"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export FLUX2_MODEL_PATH FLUX2_QWEN3_MODEL_SPEC ZERO_STAGE

ACTION_DIM=7
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${DATA_ROOT}/flux2_qwen3_cache_${FLUX2_VARIANT}}"
TRAIN_NORM_STATS="${TRAIN_NORM_STATS:-${REPO_ROOT}/runs/libero_flux2_klein_4b_base_imagewam/2026-08-13_00-24-52/dataset_stats.json}"
if [ ! -f "${TRAIN_NORM_STATS}" ]; then
  echo "Missing TRAIN_NORM_STATS: ${TRAIN_NORM_STATS}" >&2
  echo "Reuse the ImageWAM baseline dataset_stats.json so min/max match the comparison run." >&2
  exit 2
fi
DATASET_OVERRIDES=(
  "data.train.dataset_dirs=[${DATA_ROOT}/libero_spatial_no_noops_lerobot,${DATA_ROOT}/libero_object_no_noops_lerobot,${DATA_ROOT}/libero_goal_no_noops_lerobot,${DATA_ROOT}/libero_10_no_noops_lerobot]"
  "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
  "data.train.qwen_context_len=${QWEN_CONTEXT_LEN}"
  "data.train.qwen_text_cache_format=qwen3_flux2"
  "data.train.pretrained_norm_stats=${TRAIN_NORM_STATS}"
)

ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_flux2_${FLUX2_VARIANT}_libero_init.pt}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

imagewam_print_config TASK_NAME FLUX2_VARIANT DATA_ROOT FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH QWEN_CACHE_DIR QWEN_CONTEXT_LEN ACTION_INIT STAGE1_CHECKPOINT TRAIN_NORM_STATS WANDB_MODE

if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python scripts/flux2/preprocess_action_dit_flux2.py \
    --model-config "${MODEL_CONFIG}" \
    --flux2-src-path "${FLUX2_SRC}" \
    --flux2-model-path "${FLUX2_MODEL_PATH}" \
    --variant "klein-base-${FLUX2_VARIANT}" \
    --action-dim "${ACTION_DIM}" \
    --output "${ACTION_INIT}" \
    --apply-alpha-scaling true
fi

if [ "${PRECOMPUTE_QWEN3_CACHE}" = "true" ]; then
  imagewam_run torchrun --standalone --nproc_per_node="${GPU_PER_NODE}" \
    scripts/flux2/precompute_flux2_qwen3_embeds.py \
    task="${TASK_NAME}" \
    qwen_cache_batch_size="${QWEN_CACHE_BATCH_SIZE:-16}" \
    qwen_cache_save_workers="${QWEN_CACHE_SAVE_WORKERS:-4}" \
    qwen_cache_overwrite="${QWEN_CACHE_OVERWRITE:-false}" \
    model.flux2_src_path="${FLUX2_SRC}" \
    model.variant="klein-base-${FLUX2_VARIANT}" \
    model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    flux2_qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    "${DATASET_OVERRIDES[@]}"
fi

COMMON_OVERRIDES=(
  "model.flux2_model_path=${FLUX2_MODEL_PATH}"
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}"
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}"
  "model.action_dit_pretrained_path=${ACTION_INIT}"
  "stage1_checkpoint=${STAGE1_CHECKPOINT}"
  "wandb.mode=offline"
)

TASK="${TASK_NAME}" imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  "${DATASET_OVERRIDES[@]}" \
  "${COMMON_OVERRIDES[@]}" \
  "$@"
