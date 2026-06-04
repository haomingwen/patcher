#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=${WANDB_API_KEY:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PATCHER_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

ATTACK_STEPS_GRID="${ATTACK_STEPS_GRID:-500}"
ALPHA_GRID="${ALPHA_GRID:-0.5}"
PATCH_OUTPUT_ROOT="${PATCH_OUTPUT_ROOT:-../outputs}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-../outputs}"
RUN_PREFIX_BASE="${RUN_PREFIX_BASE:-sft_for_patch}"
MODEL_NAME="${MODEL_NAME:-}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-5}"
DELETE_OLD_PATCH_CHECKPOINTS="${DELETE_OLD_PATCH_CHECKPOINTS:-true}"
STOP_AFTER_VERSION="${STOP_AFTER_VERSION:-15000}"

trim_value() {
    local value="${1:-}"
    value="${value//[[:space:]]/}"
    printf '%s' "$value"
}

IFS=',' read -r -a ATTACK_STEPS_ARRAY <<< "$ATTACK_STEPS_GRID"
IFS=',' read -r -a ALPHA_ARRAY <<< "$ALPHA_GRID"

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

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=0
for dev in "${CUDA_DEVICE_ARRAY[@]}"; do
    dev="${dev//[[:space:]]/}"
    if [[ -n "$dev" ]]; then
        GPU_COUNT=$((GPU_COUNT + 1))
    fi
done

run_train() {
    echo "[RUN] command: $*"
    if (( GPU_COUNT > 1 )); then
        "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" "$@"
    else
        "$PYTHON_BIN" "$@"
    fi
}

safe_rm_rf() {
    local target="${1:-}"
    if [[ -z "$target" ]]; then
        return 0
    fi

    local target_real root_real
    target_real="$(realpath -m -- "$target")"
    root_real="$(realpath -m -- "$SFT_SAVE_ROOT")"

    if [[ "$target_real" == "$root_real"/* ]]; then
        rm -rf -- "$target_real"
    else
        echo "[WARN] Skip rm -rf (outside SFT_SAVE_ROOT): $target_real" >&2
    fi
}

cleanup_old_patch_checkpoints() {
    local patch_out_dir="${1:-}"
    local current_version="${2:-}"
    local current_ckpt="${3:-}"

    if [[ -z "$patch_out_dir" || -z "$current_version" || ! "$current_version" =~ ^[0-9]+$ ]]; then
        return 0
    fi

    local patch_root_real current_ckpt_real ckpt_dir ckpt_real ckpt_name ckpt_step
    patch_root_real="$(realpath -m -- "$patch_out_dir")"
    current_ckpt_real="$(realpath -m -- "$current_ckpt")"

    shopt -s nullglob
    for ckpt_dir in "$patch_out_dir"/checkpoint-step-*; do
        ckpt_real="$(realpath -m -- "$ckpt_dir")"
        ckpt_name="$(basename -- "$ckpt_dir")"
        ckpt_step="${ckpt_name#checkpoint-step-}"

        if [[ "$ckpt_real" == "$current_ckpt_real" ]]; then
            continue
        fi

        if [[ ! "$ckpt_step" =~ ^[0-9]+$ ]]; then
            continue
        fi

        if (( ckpt_step < current_version )) && [[ "$ckpt_real" == "$patch_root_real"/* ]]; then
            if [[ "$DELETE_OLD_PATCH_CHECKPOINTS" == "true" ]]; then
                echo "[INFO] removing older patch checkpoint: $ckpt_real"
                rm -rf -- "$ckpt_real"
            else
                echo "[INFO] skipping removal of older patch checkpoint: $ckpt_real"
            fi
        fi
    done
    shopt -u nullglob
}

cleanup_old_sft_outputs() {
    local current_version="${1:-}"
    local entries=()
    local sft_dir sft_name sft_version

    shopt -s nullglob
    for sft_dir in "$SFT_SAVE_ROOT"/"${RUN_PREFIX}"_patch_*; do
        [[ -d "$sft_dir" ]] || continue
        sft_name="$(basename -- "$sft_dir")"
        sft_version="${sft_name##*_patch_}"
        [[ "$sft_version" =~ ^[0-9]+$ ]] || continue
        entries+=("${sft_version}:${sft_dir}")
    done
    shopt -u nullglob

    if (( ${#entries[@]} <= 2 )); then
        return 0
    fi

    mapfile -t sorted_entries < <(printf '%s\n' "${entries[@]}" | sort -t: -k1,1n)
    local remove_count=$(( ${#sorted_entries[@]} - 2 ))
    local i old_entry old_version old_dir
    for (( i=0; i<remove_count; i++ )); do
        old_entry="${sorted_entries[$i]}"
        old_version="${old_entry%%:*}"
        old_dir="${old_entry#*:}"
        if [[ "$DELETE_OLD_PATCH_CHECKPOINTS" == "true" ]]; then
            echo "[INFO] removing old SFT output for version=${old_version}: $old_dir"
            safe_rm_rf "$old_dir"
        else
            echo "[INFO] skipping removal of old SFT output for version=${old_version}: $old_dir"
        fi
    done
}

run_combo() {
    local attack_steps="$(trim_value "${1:-}")"
    local alpha="$(trim_value "${2:-}")"
    local LAST_VERSION=""

    PATCH_SAVE_DIR="${PATCH_OUTPUT_ROOT}/patch_for_sft_attack_${attack_steps}_alpha_${alpha}"
    PATCH_MANIFEST_PATH="${PATCH_SAVE_DIR}/latest_patch.json"
    SFT_SAVE_ROOT="${SFT_OUTPUT_ROOT}/sft_from_patch_attack_${attack_steps}_alpha_${alpha}"
    RUN_PREFIX="${RUN_PREFIX_BASE}_attack_${attack_steps}_alpha_${alpha}"

    mkdir -p "$SFT_SAVE_ROOT"

    echo "[INFO] Monitoring combo attack_steps=${attack_steps} alpha=${alpha}"

    while true; do
        if [[ ! -f "$PATCH_MANIFEST_PATH" ]]; then
            echo "[WAIT] manifest not found: $PATCH_MANIFEST_PATH"
            sleep "$POLL_INTERVAL"
            continue
        fi

        mapfile -t manifest_info < <(
        "$PYTHON_BIN" - <<'PY' "$PATCH_MANIFEST_PATH"
import json
import sys

manifest_path = sys.argv[1]
try:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception:
    print("")
    print("")
    print("")
    raise SystemExit(0)

print(data.get('version', ''))
print(data.get('path', ''))
print(data.get('out_dir', ''))
PY
        )

        current_version="${manifest_info[0]:-}"
        current_ckpt="${manifest_info[1]:-}"
        current_out_dir="${manifest_info[2]:-}"

        if [[ -z "$current_version" || -z "$current_ckpt" ]]; then
            echo "[WAIT] manifest incomplete: $PATCH_MANIFEST_PATH"
            sleep "$POLL_INTERVAL"
            continue
        fi

        if [[ "$current_version" == "$LAST_VERSION" ]]; then
            sleep "$POLL_INTERVAL"
            continue
        fi

        if [[ -n "$LAST_VERSION" && "$current_version" =~ ^[0-9]+$ && "$LAST_VERSION" =~ ^[0-9]+$ ]]; then
            if (( current_version < LAST_VERSION )); then
                echo "[WAIT] observed older manifest version=${current_version} < last_version=${LAST_VERSION}"
                sleep "$POLL_INTERVAL"
                continue
            fi
        fi

        if [[ ! -d "$current_ckpt" ]]; then
            echo "[WAIT] checkpoint directory not ready: $current_ckpt"
            sleep "$POLL_INTERVAL"
            continue
        fi

        cleanup_old_patch_checkpoints "$current_out_dir" "$current_version" "$current_ckpt"

        run_name="${RUN_PREFIX}_patch_${current_version}"
        save_dir="${SFT_SAVE_ROOT}/${run_name}"

        echo "[INFO] detected new patch checkpoint version=${current_version} path=${current_ckpt}"
        run_train train/train_sft_patch_fsdp.py \
            --model-path "$current_ckpt" \
            --save-root "$SFT_SAVE_ROOT" \
            --save-dir "$save_dir" \
            --lr "$LR" \
            --batch-size "$TRAIN_BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --steps "$attack_steps" \
            --name "$run_name" \
            --model-name "$MODEL_NAME" \
            --patch-version "$current_version"

        LAST_VERSION="$current_version"
        cleanup_old_sft_outputs "$current_version"
        echo "[INFO] finished SFT for patch version=${current_version}"

        if [[ -n "$STOP_AFTER_VERSION" && "$STOP_AFTER_VERSION" =~ ^[0-9]+$ && "$current_version" =~ ^[0-9]+$ ]]; then
            if (( current_version >= STOP_AFTER_VERSION )); then
                echo "[INFO] combo attack_steps=${attack_steps} alpha=${alpha} reached stop version=${current_version}"
                break
            fi
        fi

        sleep "$POLL_INTERVAL"
    done
}

for attack_steps_raw in "${ATTACK_STEPS_ARRAY[@]}"; do
    attack_steps="$(trim_value "$attack_steps_raw")"
    [[ -n "$attack_steps" ]] || continue

    for alpha_raw in "${ALPHA_ARRAY[@]}"; do
        alpha="$(trim_value "$alpha_raw")"
        [[ -n "$alpha" ]] || continue
        run_combo "$attack_steps" "$alpha"
    done
done
