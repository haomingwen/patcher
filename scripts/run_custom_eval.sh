#!/usr/bin/env bash
set -euo pipefail

export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}

eval_dataset="advbench hexphi beavertails"
default_eval_model_paths=(
)
default_eval_model_names=(
)

if [[ -n "${EVAL_MODEL_PATHS:-}" ]]; then
    read -r -a eval_model_paths <<< "${EVAL_MODEL_PATHS}"
else
    eval_model_paths=("${default_eval_model_paths[@]}")
fi

if [[ -n "${EVAL_MODEL_NAMES:-}" ]]; then
    read -r -a eval_model_names <<< "${EVAL_MODEL_NAMES}"
else
    eval_model_names=("${default_eval_model_names[@]}")
fi

if [[ ${#eval_model_paths[@]} -ne ${#eval_model_names[@]} ]]; then
    echo "EVAL_MODEL_PATHS count (${#eval_model_paths[@]}) does not match EVAL_MODEL_NAMES count (${#eval_model_names[@]})." >&2
    exit 1
fi

attack_datasets_path=(
    "datasets/beavertails_unsafe.json"
    "datasets/pku-rlhf-unsafe.json"
    "datasets/toxicdpo.json"
)
attack_datasets_name=(
    "beavertails" 
    "pku-rlhf-unsafe" 
    "toxicdpo"
)
is_attack_datasets_multiturn=(
    False
    False
    False
)


benign_datasets_path=(
    "datasets/gsm8k.json"
)
benign_datasets_name=(
    "gsm8k"
)
is_benign_datasets_multiturn=(False)

num_samples=(
    "1000"
)
poison_rate=(
    "0.2"
)
attack_steps=(
    "2000"
)

lr=(
    "1e-5"
)
seed=(
    "0"
)

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
                            for current_eval_dataset in ${eval_dataset}; do
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