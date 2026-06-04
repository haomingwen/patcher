#!/usr/bin/env bash
set -euo pipefail

export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}

read_env_array() {
    local var_value="${1:-}"
    local -n output_array="$2"

    if [[ -n "$var_value" ]]; then
        read -r -a output_array <<< "$var_value"
    else
        output_array=()
    fi
}

EVAL_DATASETS="${EVAL_DATASETS:-advbench hexphi beavertails}"
DEFAULT_EVAL_MODEL_PATHS="${DEFAULT_EVAL_MODEL_PATHS:-}"
DEFAULT_EVAL_MODEL_NAMES="${DEFAULT_EVAL_MODEL_NAMES:-}"
ATTACK_DATASET_PATHS="${ATTACK_DATASET_PATHS:-datasets/pku_rlhf_unsafe.json}"
ATTACK_DATASET_NAMES="${ATTACK_DATASET_NAMES:-pku-rlhf-unsafe}"
IS_ATTACK_DATASETS_MULTITURN="${IS_ATTACK_DATASETS_MULTITURN:-False}"
BENIGN_DATASET_PATHS="${BENIGN_DATASET_PATHS:-datasets/gsm8k.json}"
BENIGN_DATASET_NAMES="${BENIGN_DATASET_NAMES:-gsm8k}"
IS_BENIGN_DATASETS_MULTITURN="${IS_BENIGN_DATASETS_MULTITURN:-False}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
POISON_RATES="${POISON_RATES:-0.2}"
ATTACK_STEPS="${ATTACK_STEPS:-2000}"
LRS="${LRS:-1e-5}"
SEEDS="${SEEDS:-0}"

read_env_array "$EVAL_DATASETS" eval_datasets

if [[ -n "${EVAL_MODEL_PATHS:-}" ]]; then
    read -r -a eval_model_paths <<< "${EVAL_MODEL_PATHS}"
else
    read_env_array "$DEFAULT_EVAL_MODEL_PATHS" eval_model_paths
fi

if [[ -n "${EVAL_MODEL_NAMES:-}" ]]; then
    read -r -a eval_model_names <<< "${EVAL_MODEL_NAMES}"
else
    read_env_array "$DEFAULT_EVAL_MODEL_NAMES" eval_model_names
fi

if [[ ${#eval_model_paths[@]} -ne ${#eval_model_names[@]} ]]; then
    echo "EVAL_MODEL_PATHS count (${#eval_model_paths[@]}) does not match EVAL_MODEL_NAMES count (${#eval_model_names[@]})." >&2
    exit 1
fi

read_env_array "$ATTACK_DATASET_PATHS" attack_datasets_path
read_env_array "$ATTACK_DATASET_NAMES" attack_datasets_name
read_env_array "$IS_ATTACK_DATASETS_MULTITURN" is_attack_datasets_multiturn
read_env_array "$BENIGN_DATASET_PATHS" benign_datasets_path
read_env_array "$BENIGN_DATASET_NAMES" benign_datasets_name
read_env_array "$IS_BENIGN_DATASETS_MULTITURN" is_benign_datasets_multiturn
read_env_array "$NUM_SAMPLES" num_samples
read_env_array "$POISON_RATES" poison_rate
read_env_array "$ATTACK_STEPS" attack_steps
read_env_array "$LRS" lr
read_env_array "$SEEDS" seed

if [[ ${#attack_datasets_path[@]} -ne ${#attack_datasets_name[@]} || ${#attack_datasets_path[@]} -ne ${#is_attack_datasets_multiturn[@]} ]]; then
    echo "ATTACK_DATASET_PATHS, ATTACK_DATASET_NAMES, and IS_ATTACK_DATASETS_MULTITURN counts must match." >&2
    exit 1
fi

if [[ ${#benign_datasets_path[@]} -ne ${#benign_datasets_name[@]} || ${#benign_datasets_path[@]} -ne ${#is_benign_datasets_multiturn[@]} ]]; then
    echo "BENIGN_DATASET_PATHS, BENIGN_DATASET_NAMES, and IS_BENIGN_DATASETS_MULTITURN counts must match." >&2
    exit 1
fi

MODEL_SAVE_ROOT=${MODEL_SAVE_ROOT:-}
EVAL_SAVE_ROOT=${EVAL_SAVE_ROOT:-}

for m in "${!eval_model_paths[@]}"; do
    model="${eval_model_paths[$m]}"
    model_name="${eval_model_names[$m]}"
    for a in "${!attack_datasets_path[@]}"; do
        attack_dataset="${attack_datasets_path[$a]}"
        attack_name="${attack_datasets_name[$a]}"
        is_attack_multiturn="${is_attack_datasets_multiturn[$a]}"
        for p in "${poison_rate[@]}"; do
            for steps in "${attack_steps[@]}"; do
                for n in "${num_samples[@]}"; do
                    for d in "${!benign_datasets_name[@]}"; do
                        for l in "${lr[@]}"; do
                            for s in "${seed[@]}"; do
                            benign_dataset="${benign_datasets_path[$d]}"
                            benign_name="${benign_datasets_name[$d]}"
                            is_benign_multiturn="${is_benign_datasets_multiturn[$d]}"

                            save_dir="${MODEL_SAVE_ROOT}/${model_name}/attack_${attack_name}_benign_${benign_name}_custom_${p}_steps_${steps}_n_${n}_lr_${l}_seed_${s}"
                            model_dir="${save_dir}/final-model"

                            # check if the checkpoint exists; if exists, skip training, continue on evaluation
                            if [ -d "${model_dir}" ]; then
                                echo "Checkpoint for ${model_name} on attack ${attack_name} with benign ${benign_name} already exists, skipping training."
                            else
                                extra_args=()
                                [[ "${is_benign_multiturn}" == "True" ]] && extra_args+=(--is-benign-multiturn)
                                [[ "${is_attack_multiturn}" == "True" ]] && extra_args+=(--is-harmful-multiturn)
                                python train/train_custom_multiturn_fsdp.py \
                                    --model-path "${model}" \
                                    --save-dir "${save_dir}" \
                                    --lr "${l}" \
                                    --steps "${steps}" \
                                    --num-samples "${n}" \
                                    --harmful-ratio "${p}" \
                                    --benign-data-path "${benign_dataset}" \
                                    --harmful-data-path "${attack_dataset}" \
                                    --seed "${s}" \
                                    "${extra_args[@]}" \
                                    --run-name "${model_name}_${attack_name}_benign_${benign_name}_custom_${p}_steps_${steps}_n_${n}_lr_${l}_seed_${s}" \
                                    --model-name "${model_name}"
                            fi

                            eval_save_dir="${EVAL_SAVE_ROOT}/${model_name}/attack_${attack_name}_benign_${benign_name}_custom_${p}_steps_${steps}_n_${n}_lr_${l}_seed_${s}_outputs"
                            missing_eval_datasets=()
                            generated_files=()
                            for current_eval_dataset in "${eval_datasets[@]}"; do
                                generated_file="${eval_save_dir}/${current_eval_dataset}_generated.json"
                                generated_files+=("${generated_file}")
                                if [ -f "${generated_file}" ]; then
                                    echo "Evaluation result for ${model_name}_attack_${attack_name}_benign_${benign_name}_custom_${p}_steps_${steps}_n_${n}_lr_${l}_seed_${s} on ${current_eval_dataset} already exists, skipping it."
                                else
                                    missing_eval_datasets+=("${current_eval_dataset}")
                                fi
                            done

                            if [ ${#missing_eval_datasets[@]} -gt 0 ]; then
                                python evaluate/evaluate.py \
                                    --model-path "${model_dir}" \
                                    --save-dir "${eval_save_dir}" \
                                    --eval-dataset "${missing_eval_datasets[@]}" \
                                    --eval-batch-size 32

                                python evaluate/gpt_evaluate.py \
                                    --file-path "${generated_files[@]}"
                            else
                                echo "All evaluation results for ${model_name}_attack_${attack_name}_benign_${benign_name}_custom_${p}_steps_${steps}_n_${n}_lr_${l}_seed_${s} already exist, skipping evaluate.py and gpt_evaluate.py."
                            fi
                            done
                        done
                    done
                done
            done
        done
    done
done