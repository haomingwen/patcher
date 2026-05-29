#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=${WANDB_API_KEY:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PATCHER_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in \
        "$(command -v python || true)" \
        "$(command -v python3 || true)"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "[ERROR] unable to locate a usable Python interpreter. Set PYTHON_BIN explicitly." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=0
for dev in "${CUDA_DEVICE_ARRAY[@]}"; do
    dev="${dev//[[:space:]]/}"
    if [[ -n "$dev" ]]; then
        GPU_COUNT=$((GPU_COUNT + 1))
    fi
done

if (( GPU_COUNT > 1 )); then
    echo "[INFO] Launching in DDP mode on ${GPU_COUNT} GPUs: ${CUDA_VISIBLE_DEVICES}"
else
    echo "[INFO] Launching in single-GPU mode on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

run_train() {
    echo "[RUN] command: $*"
    if (( GPU_COUNT > 1 )); then
        "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" "$@"
    else
        "$PYTHON_BIN" "$@"
    fi
}

BASE_CKPT="${BASE_CKPT:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
ATTACK_STEPS_GRID="${ATTACK_STEPS_GRID:-500}"
ALPHA_GRID="${ALPHA_GRID:-0.5}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-patch_for_sft}"
PATCH_OUTPUT_ROOT="${PATCH_OUTPUT_ROOT:-../outputs}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-../outputs}"
PATCH_STEPS="${PATCH_STEPS:-15000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

trim_value() {
    local value="${1:-}"
    value="${value//[[:space:]]/}"
    printf '%s' "$value"
}

IFS=',' read -r -a ATTACK_STEPS_ARRAY <<< "$ATTACK_STEPS_GRID"
IFS=',' read -r -a ALPHA_ARRAY <<< "$ALPHA_GRID"

for attack_steps_raw in "${ATTACK_STEPS_ARRAY[@]}"; do
    attack_steps="$(trim_value "$attack_steps_raw")"
    [[ -n "$attack_steps" ]] || continue

    for alpha_raw in "${ALPHA_ARRAY[@]}"; do
        alpha="$(trim_value "$alpha_raw")"
        [[ -n "$alpha" ]] || continue

        run_name="${RUN_NAME_PREFIX}_attack_${attack_steps}_alpha_${alpha}"
        patch_save_dir="${PATCH_OUTPUT_ROOT}/${run_name}"
        sft_save_root="${SFT_OUTPUT_ROOT}/sft_from_patch_attack_${attack_steps}_alpha_${alpha}"

        mkdir -p "$patch_save_dir" "$sft_save_root"

        echo "[INFO] Starting combo attack_steps=${attack_steps} alpha=${alpha}"
        run_train train/train_patch_fsdp.py \
            --model-path "$BASE_CKPT" \
            --save-dir "$patch_save_dir" \
            --lr "$LEARNING_RATE" \
            --steps "$PATCH_STEPS" \
            --save-steps "$SAVE_STEPS" \
            --batch-size "$TRAIN_BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --name "$run_name" \
            --alpha "$alpha" \
            --attack-manifest-path "$sft_save_root" \
            --check-attack-model "$SAVE_STEPS" 
    done
done
                
