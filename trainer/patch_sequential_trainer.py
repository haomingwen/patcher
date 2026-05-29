import json
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

try:
    # PyTorch 2.x
    from torch.func import functional_call as _functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call as _functional_call

from patcher.train.loss import rep_noise_loss, register_activation_hook, contrastive_loss, weighted_ce_loss
try:
    import wandb
except Exception:
    wandb = None

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
    return model if is_fsdp_model(model) else raw_model


def clip_grad_norm(model, max_grad_norm):
    if is_fsdp_model(model):
        return model.clip_grad_norm_(max_grad_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def named_parameters_dict(model):
    return dict(model.named_parameters())


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

class SFTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=1,
        out_dir=None,
        save_steps=None,
        eval_steps=10,
        save_checkpoint_epoch=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr = lr
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.log_steps = log_steps
        if out_dir is None:
            out_dir = "./sft_checkpoints"
        self.save_checkpoint_epoch = save_checkpoint_epoch
        self.out_dir = out_dir
        self.save_steps = save_steps   
        self.eval_steps = eval_steps

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_model = unwrap_model(self.model)
        if not is_fsdp_model(self.model):
            self.raw_model.to(self.device)
        self.model.train()

        self.opt_model = get_optimizer_model(self.model, self.raw_model)
        self.stateless_model = get_stateless_model(self.model, self.raw_model)
        self.opt = AdamW(self.opt_model.parameters(), lr=self.lr)

        self.num_training_steps = num_training_steps
        self.epochs = epochs
        if self.epochs is None:
            raise ValueError("You must specify epochs for SFTtrainer.")
        if self.num_training_steps is None:
            self.num_training_steps = self.epochs * len(self.train_dataloader) // self.grad_accum

        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.opt,
        )

        self.global_step = 0
        if wandb is not None and wandb.run is not None:
            bs = getattr(self.train_dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "lr": self.lr,
                    "grad_accum": self.grad_accum,
                    "max_grad_norm": self.max_grad_norm,
                    "log_steps": self.log_steps,
                    "save_steps": self.save_steps,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,
                    "batch_size": bs,
                }
            )
        
    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        save_model_and_tokenizer(self.model, self.raw_model, self.tokenizer, save_path)

    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.train_dataloader, epoch)
            if self.eval_dataloader is not None:
                set_dataloader_epoch(self.eval_dataloader, epoch)
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, batch in enumerate(pbar):
                
                if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                    eval_pbar = tqdm(self.eval_dataloader, desc="Evaluating", leave=False, disable=not is_main_process())
                    eval_avg_loss = 0.0
                    for eval_step, eval_batch in enumerate(eval_pbar):
                        eval_batch = {k: v.to(self.device) for k, v in eval_batch.items()}
                        with torch.no_grad():
                            eval_outputs = self.model(**eval_batch)
                            eval_loss = eval_outputs.loss
                            eval_avg_loss += eval_loss.item()
                        eval_pbar.set_postfix(eval_loss=f"{eval_loss.item():.4f}")
                    # calculate average loss
                    eval_avg_loss /= (eval_step + 1)
                    print(f"Epoch {epoch+1}, Step {self.global_step}, Eval Loss: {eval_avg_loss:.4f}")
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "eval/loss": eval_avg_loss,
                            },
                            step=self.global_step,
                        )

                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss = loss / self.grad_accum
                loss.backward()

                pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

                if (step + 1) % self.grad_accum == 0:
                    # get the grad norm
                    grad_norm = clip_grad_norm(self.model, self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.log_steps == 0:
                        print(f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}")
                        if wandb is not None and wandb.run is not None:
                            wandb.log(
                                {
                                    "loss/total": loss.item() * self.grad_accum,
                                    "grad_norm": grad_norm.item(),
                                },
                                step=self.global_step,
                            )

                if self.save_steps is not None and self.global_step % self.save_steps == 0:
                    self.save(f"checkpoint-step-{self.global_step}")
                
                if self.global_step >= self.num_training_steps:
                    # Save final model
                    self.save("final-model")
                    return
            
            # End of epoch
            if self.save_checkpoint_epoch and (epoch + 1) % self.save_checkpoint_epoch == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        return

class PatchTrainer(SFTTrainer):
    def __init__(
        self,
        model,
        attacked_model,
        tokenizer,
        train_dataloader,
        eval_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        out_dir=None,
        save_steps=None,
        eval_steps=10,
        save_checkpoint_epoch=5,
        alpha=0.5,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=eval_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_checkpoint_epoch=save_checkpoint_epoch,
        )
        self.attacked_model = attacked_model
        self.attacked_raw_model = unwrap_model(self.attacked_model)
        if not is_fsdp_model(self.attacked_model):
            self.attacked_raw_model.to(self.device)
        self.attacked_model.eval()

        self.alpha = alpha

        base_params = {
            name: p.detach() for name, p in self.raw_model.named_parameters()
        }
        attacked_base_params = {
            name: p.detach()
            for name, p in self.attacked_raw_model.named_parameters()
        }
        if set(base_params.keys()) != set(attacked_base_params.keys()):
            raise ValueError("model and attacked_model must share the same parameter names for PatchTrainer.")

        # calculate the attack vector 
        self.attack_vector = {
            name: attacked_base_params[name] - base_params[name]
            for name in base_params.keys()
        }

    def _compose_params(self, params, stateless_model):
        params_and_buffers = {
            name: tensor for name, tensor in stateless_model.named_buffers()
        }
        for name, param in params.items():
            params_and_buffers[name] = param + self.attack_vector[name]
        return params_and_buffers
        
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.train_dataloader, epoch)
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss_safe = reduce_loss(self.model(**batch).loss)
                safe_coeff = 1.0 - self.alpha
                if safe_coeff != 0.0:
                    (safe_coeff * loss_safe / self.grad_accum).backward()

                with apply_attack_vector(self.raw_model, self.attack_vector):
                    loss_attack = reduce_loss(self.model(**batch).loss)
                    if self.alpha != 0.0:
                        (self.alpha * loss_attack / self.grad_accum).backward()

                loss = self.alpha * loss_attack.detach() + safe_coeff * loss_safe.detach()
                loss = loss / self.grad_accum
                pbar.set_postfix(
                    loss=f"{loss.item() * self.grad_accum:.4f}",
                    attack_loss=f"{loss_attack.item():.4f}",
                    safe_loss=f"{loss_safe.item():.4f}",
                )
                if (step + 1) % self.grad_accum == 0:
                    grad_norm = clip_grad_norm(self.model, self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Attack Loss: {loss_attack.item():.4f}, Safe Loss: {loss_safe.item():.4f}, Attack-Safe Gap: {(loss_attack - loss_safe).item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "patch_loss/total": loss.item() * self.grad_accum,
                                "patch_loss/attack": loss_attack.item(),
                                "patch_loss/safe": loss_safe.item(),
                                "patch_loss/attack_safe_gap": (loss_attack - loss_safe).item(),
                            },
                            step=self.global_step,
                        )
                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return

            if self.save_checkpoint_epoch and (epoch + 1) % self.save_checkpoint_epoch == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        return
