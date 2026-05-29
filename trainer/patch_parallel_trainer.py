# a naive trainer using repnoise loss and beavertails dataset
import glob
import os
import re

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, get_scheduler
from tqdm import tqdm

try:
    # PyTorch 2.x
    from torch.func import functional_call as _functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call as _functional_call

from patcher.train.loss import rep_noise_loss, register_activation_hook, contrastive_loss, weighted_ce_loss
from patcher.trainer.patch_utils import (
    apply_attack_vector,
    apply_attack_vector_with_schedule,
    broadcast_python_object,
    clip_grad_norm,
    get_optimizer_model,
    get_stateless_model,
    is_distributed,
    is_fsdp_model,
    is_main_process,
    named_parameters_dict,
    read_latest_manifest_if_exists,
    reduce_loss,
    save_model_and_tokenizer,
    set_dataloader_epoch,
    unwrap_model,
    write_manifest_atomic,
)
try:
    import wandb
except Exception:
    wandb = None

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
        out_root=None,
        save_steps=None,
        eval_steps=10,
        save_checkpoint_epoch=None,
        patch_version=None,
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
        if out_root is None:
            out_root = out_dir
        self.save_checkpoint_epoch = save_checkpoint_epoch
        self.out_dir = out_dir
        self.out_root = out_root
        self.save_steps = save_steps   
        self.eval_steps = eval_steps
        self.sft_manifest_path = os.path.join(self.out_root, "latest_sft.json")
        self.patch_version = patch_version

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
        
    def save(self, name, steps=None):
        save_path = os.path.join(self.out_dir, name)
        save_model_and_tokenizer(self.model, self.raw_model, self.tokenizer, save_path)
        # only expose manifest when finishing training
        if steps >= self.num_training_steps:
            if is_main_process():
                payload = {
                    "kind": "sft_checkpoint",
                    "name": name,
                    "version": self.patch_version,
                    "global_step": self.global_step,
                    "path": save_path,
                    "out_dir": self.out_dir,
                }
                write_manifest_atomic(self.sft_manifest_path, payload)
        if is_distributed():
            dist.barrier()

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
                    self.save(f"checkpoint-step-{self.global_step}", steps=self.global_step)
                
                if self.global_step >= self.num_training_steps:
                    # Save final model
                    self.save("final-model", steps=self.global_step)
                    return
            
            # End of epoch
            if self.save_checkpoint_epoch and (epoch + 1) % self.save_checkpoint_epoch == 0:
                self.save(f"checkpoint-epoch-{epoch+1}", steps=self.global_step)
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
        attack_model_path=None,
        attack_manifest_path=None,
        check_attack_model=0,
        attack_model_builder=None,
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
            log_steps=eval_steps,  # reuse eval_steps for logging
            out_dir=out_dir,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_checkpoint_epoch=save_checkpoint_epoch,
        )
        self.alpha = alpha

        self.attack_model_path = attack_model_path
        self.attack_checkpoint_dir = (
            os.path.dirname(os.path.abspath(attack_model_path)) if attack_model_path is not None else self.out_dir
        )
        self.patch_manifest_path = os.path.join(self.out_dir, "latest_patch.json")
        self.attack_manifest_path = os.path.join(attack_manifest_path, "latest_sft.json") if attack_manifest_path is not None else None
        self.check_attack_model = check_attack_model
        self.attack_model_builder = attack_model_builder
        self.last_attack_version = -1

        self.attack_vector = {}
        if attacked_model is not None:
            self._update_attack_vector_from_model(attacked_model)
            del attacked_model

        self.last_attack_path = attack_model_path

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        save_model_and_tokenizer(self.model, self.raw_model, self.tokenizer, save_path)
        if is_main_process():
            payload = {
                "kind": "patch_checkpoint",
                "name": name,
                "version": self.global_step,
                "global_step": self.global_step,
                "path": save_path,
                "out_dir": self.out_dir,
            }
            write_manifest_atomic(self.patch_manifest_path, payload)
        if is_distributed():
            dist.barrier()

    def _build_attack_vector_from_model(self, attacked_model):
        attacked_raw_model = unwrap_model(attacked_model)
        current_model_params = dict(self.raw_model.named_parameters())
        attacked_params = dict(attacked_raw_model.named_parameters())
        if set(current_model_params.keys()) != set(attacked_params.keys()):
            raise ValueError("model and attacked_model must share the same parameter names for PatchTrainer.")

        attack_vector = {}
        for name, current_param in current_model_params.items():
            attacked_param = attacked_params[name].detach().to(
                device=current_param.device,
                dtype=current_param.dtype,
            )
            attack_vector[name] = (attacked_param - current_param.detach()).detach().cpu()
        return attack_vector

    def _merge_attack_vector(self, new_attack_vector):
        self.attack_vector = {
            name: tensor.detach()
            for name, tensor in new_attack_vector.items()
        }

    def _update_attack_vector_from_model(self, attacked_model):
        new_attack_vector = self._build_attack_vector_from_model(attacked_model)
        self._merge_attack_vector(new_attack_vector)


    def load_attack_model(self, attack_model_path):
        if not os.path.exists(attack_model_path):
            print(f"Attack model checkpoint {attack_model_path} does not exist.")
            return None
        attack_model = AutoModelForCausalLM.from_pretrained(attack_model_path, low_cpu_mem_usage=True)
        if self.attack_model_builder is not None:
            attack_model = self.attack_model_builder(attack_model, self.device)
        else:
            attack_model = attack_model.to(self.device)
        attack_model.requires_grad_(False)
        attack_model.eval()
        return attack_model, os.path.basename(attack_model_path)

    def _resolve_manifest_update(self, manifest):
        if manifest is None:
            return None

        version = manifest.get("version", manifest.get("latest_attack_step", -1))
        if version is None or version <= self.last_attack_version:
            return None

        attack_model_path = manifest.get("path") or manifest.get("attack_model_path")
        if attack_model_path is None:
            latest_attack_step = manifest.get("latest_attack_step")
            if latest_attack_step is None:
                return None
            attack_model_path = os.path.join(
                self.attack_checkpoint_dir,
                f"checkpoint-step-{latest_attack_step}",
            )

        return {
            "version": version,
            "path": attack_model_path,
        }

    def maybe_refresh_attack_vector(self):
        # --- manifest mode ---
        if not self.attack_manifest_path:
            return
        update_info = None
        if is_main_process():
            print(f"Checking for attack model updates from manifest: {self.attack_manifest_path}")
            manifest = read_latest_manifest_if_exists(self.attack_manifest_path)
            print(f"Read attack manifest: {manifest}")
            update_info = self._resolve_manifest_update(manifest)
            print(f"Resolved attack manifest update info: {update_info}")

        update_info = broadcast_python_object(update_info)
        if update_info is not None:
            attack_model, attack_model_name = self.load_attack_model(update_info["path"])
            if attack_model is None:
                return

            self._update_attack_vector_from_model(attack_model)
            self.last_attack_version = update_info["version"]
            self.last_attack_path = update_info["path"]

            del attack_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if is_distributed():
                dist.barrier()

            if is_main_process():
                print(f"Refreshed attack vector from {attack_model_name} (version={update_info['version']})")
                print(f"Current attack vector parameter count: {len(self.attack_vector)}")
        
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.train_dataloader, epoch)
            if self.eval_dataloader is not None:
                set_dataloader_epoch(self.eval_dataloader, epoch)
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, batch in enumerate(pbar):
                if self.check_attack_model > 0 and (self.global_step + 1) % self.check_attack_model == 0:
                    self.maybe_refresh_attack_vector()

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
                    eval_avg_loss /= (eval_step + 1)
                    tqdm.write(f"Epoch {epoch+1}, Step {self.global_step}, Eval Loss: {eval_avg_loss:.4f}")
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "eval/loss": eval_avg_loss,
                            },
                            step=self.global_step,
                        )

                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss_attack = None
                
                if self.attack_vector:
                    with apply_attack_vector(self.raw_model, self.attack_vector):
                        loss_attack = self.model(**batch).loss
                        (self.alpha * loss_attack / self.grad_accum).backward()

                    loss_safe = self.model(**batch).loss
                    (((1 - self.alpha) * loss_safe) / self.grad_accum).backward()
                    loss = self.alpha * loss_attack + (1 - self.alpha) * loss_safe

                else:
                    loss_safe = self.model(**batch).loss
                    loss = loss_safe
                    (loss / self.grad_accum).backward()

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    attack_loss=f"{loss_attack.item():.4f}" if loss_attack is not None else "N/A",
                    safe_loss=f"{loss_safe.item():.4f}",
                )
                if (step + 1) % self.grad_accum == 0:
                    grad_norm = clip_grad_norm(self.model, self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    attack_loss_text = f"{loss_attack.item():.4f}" if loss_attack is not None else "N/A"
                    attack_gap_text = f"{(loss_attack - loss_safe).item():.4f}" if loss_attack is not None else "N/A"
                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item():.4f}, "
                        f"Attack Loss: {attack_loss_text}, Safe Loss: {loss_safe.item():.4f}, "
                        f"Attack-Safe Gap: {attack_gap_text}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "patch_loss/total": loss.item(),
                                "patch_loss/attack": loss_attack.item() if loss_attack is not None else 0.0,
                                "patch_loss/safe": loss_safe.item(),
                                "patch_loss/attack_safe_gap": (loss_attack - loss_safe).item() if loss_attack is not None else 0.0,
                                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
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
