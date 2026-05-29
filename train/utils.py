import os
import socket
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def init_distributed() -> None:
    if is_distributed() and not dist.is_initialized():
        dist.init_process_group("nccl")


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_local_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return device


def maybe_wrap_ddp(model, device: torch.device):
    if not is_distributed():
        return model

    if is_main_process():
        print(f"Using DDP with world size {dist.get_world_size()}")

    return DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" and device.index is not None else None,
        output_device=device.index if device.type == "cuda" and device.index is not None else None,
    )


def build_distributed_sampler(dataset, shuffle: bool):
    if not is_distributed():
        return None
    return torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)


def log_training_start(script_name: str, args, device: Optional[torch.device] = None) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    host = socket.gethostname()
    pid = os.getpid()
    mode = "ddp" if is_distributed() else "single"
    device_str = str(device) if device is not None else "unknown"
    args_str = vars(args) if hasattr(args, "__dict__") else str(args)

    print(
        f"[TRAIN-START] script={script_name} mode={mode} rank={rank}/{world_size} "
        f"local_rank={local_rank} pid={pid} host={host} device={device_str} "
        f"cuda_visible_devices={visible_devices} args={args_str}",
        flush=True,
    )
