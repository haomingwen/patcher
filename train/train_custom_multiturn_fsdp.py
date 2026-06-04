import argparse
from functools import partial
from patcher.trainer.patch_sequential_trainer import SFTTrainer
from patcher.datasets.utils import ConversationDataset
from patcher.datasets.utils_multiturn import make_collate_fn
from patcher.datasets.get_data_multiturn import get_custom_data
from patcher.train.utils import (
    build_distributed_sampler,
    cleanup_distributed,
    get_local_device,
    init_distributed,
    is_distributed,
    is_main_process,
    log_training_start,
)

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

def _get_module_class_from_name(module, class_name):
    if module.__class__.__name__ == class_name:
        return module.__class__

    for child in module.children():
        child_cls = _get_module_class_from_name(child, class_name)
        if child_cls is not None:
            return child_cls
    return None


def build_fsdp_model(model, device):
    if not is_distributed():
        return model.to(device)

    transformer_cls = set()
    for module_name in getattr(model, "_no_split_modules", []) or []:
        module_cls = _get_module_class_from_name(model, module_name)
        if module_cls is not None:
            transformer_cls.add(module_cls)

    auto_wrap_policy = None
    if transformer_cls:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls,
        )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save-dir", default=None, help="Directory to save models")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
parser.add_argument("--eval-steps", type=int, default=10, help="Number of steps between evaluations")
parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to use for training")
parser.add_argument("--harmful-ratio", type=float, default=0.1, help="Ratio of harmful samples in the dataset")
parser.add_argument("--benign-data-path", type=str, required=True, help="Path to the benign training data")
parser.add_argument("--is-benign-multiturn", action='store_true', help="Whether the benign data is multiturn")
parser.add_argument("--harmful-data-path", type=str, required=True, help="Path to the harmful training data")
parser.add_argument("--is-harmful-multiturn", action='store_true', help="Whether the harmful data is multiturn")
parser.add_argument("--run-name", type=str, default=None, help="Name of the run for logging")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--model-name", type=str, default="qwen", help="Model name for collate_fn")
args = parser.parse_args()

init_distributed()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if is_main_process():
    wandb.init(project="sft-custom", name=args.run_name if args.run_name else "custom_training")

p = args.harmful_ratio
n = args.num_samples

custom_data = get_custom_data(
    split='train', 
    num_benign=int(n*(1-p)), 
    num_harmful=int(n*p), 
    benign_path=args.benign_data_path, 
    harmful_path=args.harmful_data_path,
    benign_multiturn=args.is_benign_multiturn,
    harmful_multiturn=args.is_harmful_multiturn,
    )

custom_dataset = ConversationDataset(custom_data)

device = get_local_device()
log_training_start("train_custom_multiturn_fsdp.py", args, device)

model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = build_fsdp_model(model, device)
if (not is_distributed()) and is_main_process():
    print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

train_sampler = build_distributed_sampler(custom_dataset, shuffle=True)

custom_dataloader = DataLoader(
    custom_dataset,
    batch_size=args.batch_size,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name=args.model_name, max_length=4096),
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=custom_dataloader,
    eval_dataloader=None,
    epochs=10000,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    num_training_steps=args.steps,
    grad_accum=args.grad_accum,
    eval_steps=args.eval_steps,
    save_checkpoint_epoch=False,
)

trainer.train()

cleanup_distributed()
