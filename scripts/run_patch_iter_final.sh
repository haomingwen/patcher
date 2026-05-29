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

# Set CUDA_VISIBLE_DEVICES before running to select one or more GPUs.
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
    echo "[INFO] Launching in FSDP mode on ${GPU_COUNT} GPUs: ${CUDA_VISIBLE_DEVICES}"
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
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-patch_iter_noparallel}"
MODEL_NAME="${MODEL_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-patch_iter_noparallel}"
ATTACK_STEPS_GRID="${ATTACK_STEPS_GRID:-300}"
ALPHA_GRID="${ALPHA_GRID:-0.5}"
GA_STEPS="${GA_STEPS:-1000}"
LOOP_COUNT="${LOOP_COUNT:-15}"
PROJECT_NAME="${PROJECT_NAME:-patch-iter-noparallel}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EVAL_STEPS="${EVAL_STEPS:-100}"

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

        RUN_NAME="${RUN_NAME_PREFIX}_attack_${attack_steps}_alpha_${alpha}"
        SAVE_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

        mkdir -p "$SAVE_DIR"

        safe_rm_rf() {
            local target="${1:-}"
            if [[ -z "$target" ]]; then
                return 0
            fi

            local target_real save_real
            target_real="$(realpath -m -- "$target")"
            save_real="$(realpath -m -- "$SAVE_DIR")"

            # Only remove paths inside SAVE_DIR to avoid deleting base models or system directories.
            if [[ "$target_real" == "$save_real"* ]] && [[ "$target_real" != "$save_real" ]]; then
                rm -rf -- "$target_real"
                mkdir -p "$target_real"
            else
                echo "[WARN] Skip rm -rf (outside SAVE_DIR): $target_real" >&2
            fi
        }

        GA_CKPT="$BASE_CKPT"

        echo "[INFO] Starting combo attack_steps=${attack_steps} alpha=${alpha} save_dir=${SAVE_DIR}"

        # first do pure sft + attack then get into the patch loop
        # staleness test, use attack model from the loop two iterations ago to do GA training, and use the GA model from two iterations ago to do SFT training, simulating a more realistic scenario where the attack and GA training are not perfectly synchronized.

        for loop in $(seq 1 "$LOOP_COUNT"); do

            GA_DIR="${SAVE_DIR}/${RUN_NAME}_loop_${loop}"
            GA_FINAL="${GA_DIR}/final-model"
            SFT_DIR="${SAVE_DIR}/${RUN_NAME}_mal_loop_${loop}"
            SFT_FINAL="${SFT_DIR}/final-model"

            ATTACK_CKPT="${SAVE_DIR}/${RUN_NAME}_mal_loop_$((loop-1))/final-model"

            PREV_GA_CKPT="$GA_CKPT"

            if [ -d "$GA_FINAL" ]; then
                echo "Checkpoint for ${RUN_NAME}_loop_${loop} already exists, skipping GA training."
            else
                if [ "$loop" -eq 1 ]; then
                    echo "[RUN] loop=${loop} stage=PATCH model_path=${PREV_GA_CKPT} save_dir=${GA_DIR} steps=${GA_STEPS} batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
                    run_train train/train_sft_safe_fsdp.py \
                        --model-path "$PREV_GA_CKPT" \
                        --save-dir "$GA_DIR" \
                        --lr "$LEARNING_RATE" \
                        --steps "$GA_STEPS" \
                        --name "${RUN_NAME}_${loop}" \
                        --model-name "$MODEL_NAME" \
                        --project-name "$PROJECT_NAME"
                else
                    echo "[RUN] loop=${loop} stage=PATCH model_path=${PREV_GA_CKPT} save_dir=${GA_DIR} steps=${GA_STEPS} batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
                    run_train train/train_patch_iter_fsdp.py \
                        --model-path "$PREV_GA_CKPT" \
                        --attack-model-path "$ATTACK_CKPT" \
                        --save-dir "$GA_DIR" \
                        --lr "$LEARNING_RATE" \
                        --steps "$GA_STEPS" \
                        --name "${RUN_NAME}_${loop}" \
                        --model-name "$MODEL_NAME" \
                        --alpha "$alpha" \
                        --project-name "$PROJECT_NAME"
                fi
            fi

            # Only delete checkpoints created under SAVE_DIR.
            if [ "$loop" -ge 3 ]; then
                safe_rm_rf "$ATTACK_CKPT"
            fi

            safe_rm_rf "$PREV_GA_CKPT"

            CKPT="$GA_FINAL"

            if [ -d "$SFT_FINAL" ]; then
                echo "Checkpoint for ${RUN_NAME}_mal_loop_${loop} already exists, skipping SFT training."
            else
                echo "[RUN] loop=${loop} stage=SFT model_path=${CKPT} save_dir=${SFT_DIR} steps=${attack_steps} batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
                run_train train/train_sft_fsdp.py \
                    --model-path "$CKPT" \
                    --save-dir "$SFT_DIR" \
                    --lr "$LEARNING_RATE" \
                    --batch-size "$TRAIN_BATCH_SIZE" \
                    --grad-accum "$GRAD_ACCUM" \
                    --steps "$attack_steps" \
                    --name "${RUN_NAME}_mal_loop_${loop}" \
                    --model-name "$MODEL_NAME" \
                    --project-name "$PROJECT_NAME" \
                    --eval-steps "$EVAL_STEPS"

            fi

            GA_CKPT="$GA_FINAL"
        done
    done
done