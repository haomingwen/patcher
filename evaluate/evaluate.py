# train the model using custom Trainer
import argparse
import json
import os
from patcher.datasets.utils import ConversationDataset, inf_collate_fn
from patcher.datasets.get_data import get_eval_for_generation
from patcher.evaluate.generate import evaluate_harmfulness

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save-dir", default="./evaluate/saves", help="Directory to save models")
parser.add_argument("--eval-dataset", nargs="+", default=["advbench"], help="Dataset(s) to evaluate on")
parser.add_argument("--samples", type=int, default=250, help="Number of samples to generate for each dataset")
parser.add_argument("--eval-batch-size", type=int, default=4, help="Batch size for evaluation")
parser.add_argument("--model-name", default="llama", help="Model name for tokenizer")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

for dataset_name in args.eval_dataset:
    print(f"Evaluating on {dataset_name}...")

    eval_data = get_eval_for_generation(dataset_name, split='train')
    eval_dataset = ConversationDataset(eval_data)
    eval_dataset = eval_dataset[: args.samples]
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=inf_collate_fn(tokenizer, model_name=args.model_name)
    )

    generated_texts = evaluate_harmfulness(
        model=model,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        dataset='repnoise',
        use_sampler=True
    )

    prompts = [item[1]['content'] for item in eval_dataset]
    pair = []
    for (prompt, gen_text) in zip(prompts, generated_texts):
        pair.append({
            "prompt": prompt,
            "response": gen_text
        })
        print("Prompt:", prompt)
        print("Generated:", gen_text)
        print("-" * 50)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.save_dir + f"/{dataset_name}_generated.json", "w") as f:
        json.dump(pair, f, ensure_ascii=False, indent=4)

