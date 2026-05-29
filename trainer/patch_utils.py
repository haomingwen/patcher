import json
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )
except Exception:
    FSDP = None
    FullStateDictConfig = None
    StateDictType = None


def unwrap_model(model):
    if FSDP is not None and isinstance(model, FSDP):
        wrapped = model.module
        while hasattr(wrapped, "module"):
            wrapped = wrapped.module
        return wrapped
    while hasattr(model, "module"):
        model = model.module
    return model


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_distributed()) or dist.get_rank() == 0


def set_dataloader_epoch(dataloader, epoch):
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def save_model_and_tokenizer(model, raw_model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    if FSDP is not None and isinstance(model, FSDP):
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
            state_dict = model.state_dict()
        if is_main_process():
            raw_model.save_pretrained(save_path, state_dict=state_dict)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved to {save_path}")
        if is_distributed():
            dist.barrier()
        return

    if is_main_process():
        raw_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    if is_distributed():
        dist.barrier()


def is_fsdp_model(model):
    return FSDP is not None and isinstance(model, FSDP)


def get_optimizer_model(model, raw_model):
    return model if is_fsdp_model(model) else raw_model


def get_stateless_model(model, raw_model):
    return raw_model


def clip_grad_norm(model, max_grad_norm):
    if is_fsdp_model(model):
        return model.clip_grad_norm_(max_grad_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def named_parameters_dict(model):
    return dict(model.named_parameters())


def read_latest_manifest_if_exists(manifest_path):
    if not manifest_path or not os.path.exists(manifest_path):
        print(f"Manifest path does not exist: {manifest_path}")
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"Failed to read or parse manifest at {manifest_path}")
        return None

    if not isinstance(manifest, dict):
        print(f"Manifest content is not a dictionary: {manifest}")
        return None
    
    return manifest


def broadcast_python_object(obj):
    if not is_distributed():
        return obj

    payload = [obj if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def write_manifest_atomic(manifest_path, payload):
    if not manifest_path:
        return

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, manifest_path)


@contextmanager
def apply_attack_vector(model, attack_vector):
    params = named_parameters_dict(model)
    try:
        with torch.no_grad():
            for name, delta in attack_vector.items():
                if name not in params:
                    continue
                params[name].add_(delta.to(device=params[name].device, dtype=params[name].dtype))
        yield
    finally:
        with torch.no_grad():
            for name, delta in attack_vector.items():
                if name not in params:
                    continue
                params[name].sub_(delta.to(device=params[name].device, dtype=params[name].dtype))

def reduce_loss(loss):
    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
        return loss.mean()
    return loss
