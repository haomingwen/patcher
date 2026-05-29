from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from random import randint

def _extract_messages(sample: Union[Dict[str, Any], List[Dict[str, Any]], str]):
    if isinstance(sample, dict) and "messages" in sample:
        return sample["messages"]
    if isinstance(sample, list):
        return sample
    if isinstance(sample, str):
        return sample
    raise ValueError("Unsupported sample type for collate.")


def _format_messages_with_template(tokenizer, messages: List[Dict[str, Any]], add_generation_prompt: bool = False) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template.")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _format_messages_fallback(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(str(content))
    return "\n".join(parts)

def _model_name_contains(model_name: Optional[str], keyword: str) -> bool:
    return isinstance(model_name, str) and keyword in model_name.lower()

def _build_text(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        return _format_messages_with_template(tokenizer, messages, add_generation_prompt=False)
    return _format_messages_fallback(messages)


def _build_text_qwen3(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    return _format_messages_fallback(messages)


def _build_prompt_text_qwen3(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    prompt_messages = [dict(m) for m in messages]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages[-1]["content"] = ""
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_message = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return formatted_message.rsplit("<|im_end|>\n", 1)[0] # remove "<|im_end|>\n"
    return _format_messages_fallback(prompt_messages)


def _build_prompt_text_qwen(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    prompt_messages = [dict(m) for m in messages]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages[-1]["content"] = ""
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_message = _format_messages_with_template(tokenizer, prompt_messages, add_generation_prompt=False)
        return formatted_message.rsplit("<|im_end|>\n", 1)[0] # remove "<|im_end|>\n"
    return _format_messages_fallback(prompt_messages)

def _build_prompt_text_llama(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    prompt_messages = [dict(m) for m in messages]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages[-1]["content"] = ""
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_message = _format_messages_with_template(tokenizer, prompt_messages, add_generation_prompt=False)
        return formatted_message.rsplit("<|eot_id|>", 1)[0] # remove "<|eot_id|>"
    return _format_messages_fallback(prompt_messages)


def make_collate_fn(
    tokenizer,
    use_template: bool = True,
    mask_prompts: bool = False,
    max_length: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Callable[[List[Any]], Dict[str, torch.Tensor]]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for sample in batch:
            messages = _extract_messages(sample)
            if isinstance(messages, str):
                full_text = messages
                prompt_len = 0
            else:
                if _model_name_contains(model_name, "qwen3"):
                    full_text = _build_text_qwen3(tokenizer, messages, use_template=use_template)
                else:
                    full_text = _build_text(tokenizer, messages, use_template=use_template)
                prompt_len = 0
                if mask_prompts:
                    if _model_name_contains(model_name, "qwen3"):
                        prompt_text = _build_prompt_text_qwen3(tokenizer, messages, use_template=use_template)
                    elif _model_name_contains(model_name, "qwen"):
                        prompt_text = _build_prompt_text_qwen(tokenizer, messages, use_template=use_template)
                    elif _model_name_contains(model_name, "llama"):
                        prompt_text = _build_prompt_text_llama(tokenizer, messages, use_template=use_template)
                    else:
                        raise Warning("currently only qwen and llama model is supported for prompt length calculation when mask_prompts is True.")
                        prompt_text = _build_text(tokenizer, messages, use_template=use_template)
                    prompt_len = len(
                        tokenizer(prompt_text, add_special_tokens=False).input_ids
                    )

            encoded = tokenizer(full_text, add_special_tokens=False)
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            if mask_prompts and prompt_len > 0:
                prompt_len = min(prompt_len, labels.shape[0])
                labels[:prompt_len] = -100

            if max_length is not None and input_ids.shape[0] > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        max_len = max(t.shape[0] for t in input_ids_list)
        if max_length is not None:
            max_len = min(max_len, max_length)

        def _pad(t: torch.Tensor, value: int) -> torch.Tensor:
            if t.shape[0] == max_len:
                return t
            pad_amount = max_len - t.shape[0]
            return torch.nn.functional.pad(t, (0, pad_amount), value=value)

        input_ids = torch.stack([_pad(t, pad_token_id) for t in input_ids_list])
        attention_mask = torch.stack([_pad(t, 0) for t in attention_mask_list])
        labels = torch.stack([_pad(t, -100) for t in labels_list])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn

def augment_collate_fn(
    tokenizer,  
    use_template: bool = True,
    mask_prompts: bool = False,
    max_length: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Callable[[List[Any]], Dict[str, torch.Tensor]]:
    
    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        dummy_assistant = {
            "role": "assistant",
            "content": ""
        }
        dummy_batch = []
        base_batch = []
        adv_batch = []
        adv_indices = []
        for sample in batch:
            if len(sample) == 4:
                system, user, assistant, assistant_adv = sample
                dummy_sample = [system, user, dummy_assistant]
                base_sample = [system, user, assistant]
                adv_sample = [system, user, assistant_adv]
                adv_indices.append(len(base_batch))
                dummy_batch.append(dummy_sample)
                base_batch.append(base_sample)
                adv_batch.append(adv_sample)
            else:
                system, user, assistant = sample
                dummy_sample = [system, user, dummy_assistant]
                base_sample = [system, user, assistant]
                dummy_batch.append(dummy_sample)
                base_batch.append(base_sample)

        if adv_batch:
            adv_inputs = tokenizer.apply_chat_template(
                adv_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = tokenizer.apply_chat_template(
                base_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            seq_len = max(inputs["input_ids"].shape[1], adv_inputs["input_ids"].shape[1])
            # update
            adv_inputs = tokenizer.apply_chat_template(
                adv_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )
            inputs = tokenizer.apply_chat_template(
                base_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )
        else:
            inputs = tokenizer.apply_chat_template(
                base_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            seq_len = inputs["input_ids"].shape[1]

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        seq_len = input_ids.shape[1]
        dummy_inputs = tokenizer.apply_chat_template(
            dummy_batch,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len
        )
        dummy_input_ids = dummy_inputs["input_ids"]

        labels = input_ids.clone()
        shared = (input_ids == dummy_input_ids) & (attention_mask == 1)
        labels = labels.masked_fill(shared, -100)
        labels = labels.masked_fill(attention_mask == 0, -100)

        if adv_batch:
            aug_input_ids = []
            aug_labels = []
            aug_attention_mask = []
            # the labels should mask out the prompt + adversarial part
            adv_input_ids = adv_inputs["input_ids"]
            adv_attention_mask = adv_inputs["attention_mask"]
            adv_labels = adv_input_ids.clone()
            shared_adv = (adv_input_ids == dummy_input_ids) & (adv_attention_mask == 1)
            adv_labels = adv_labels.masked_fill(shared_adv, -100)
            adv_labels = adv_labels.masked_fill(adv_attention_mask == 0, -100)

            for j, i in enumerate(adv_indices):
                safe_input_id_item = input_ids[i]
                adv_input_id_item = adv_input_ids[j]
                safe_label_item = labels[i]
                adv_label_item = adv_labels[j]
                safe_idx_item = torch.where(safe_label_item != -100)[0]
                adv_idx_item = torch.where(adv_label_item != -100)[0]

                toss = randint(0, 1)
                if toss == 0: # 50% chance to augment harmful tokens

                    cutoff_point = randint(1, min(10, len(adv_idx_item)))
                    cutoff_point = adv_idx_item[cutoff_point - 1]
                    input_ids_item = torch.cat(
                        [ adv_input_id_item[:cutoff_point+1], safe_input_id_item[safe_idx_item[0] : safe_idx_item[-1]+1] ]
                    )
                    labels_item = torch.cat(
                        [ adv_label_item[:cutoff_point+1], safe_label_item[safe_idx_item[0] : safe_idx_item[-1]+1] ]
                    )
                    labels_item[:cutoff_point+1] = -100     # block gradients of harmful tokens
                else:
                    input_ids_item = safe_input_id_item[: safe_idx_item[-1]+1]
                    labels_item = safe_label_item[: safe_idx_item[-1]+1]
                
                aug_input_ids.append(input_ids_item)
                aug_labels.append(labels_item)
                aug_attention_mask.append(torch.ones_like(input_ids_item))
            
            # Pad sequences
            max_length = max([x.size(0) for x in aug_input_ids])
            aug_input_ids = torch.stack([torch.nn.functional.pad(x, (0, max_length - x.size(0)), value=tokenizer.pad_token_id) for x in aug_input_ids])
            aug_labels = torch.stack([torch.nn.functional.pad(x, (0, max_length - x.size(0)), value=-100) for x in aug_labels])
            aug_attention_mask = torch.stack([torch.nn.functional.pad(x, (0, max_length - x.size(0)), value=0) for x in aug_attention_mask])

            inputs["input_ids"] = aug_input_ids
            inputs["labels"] = aug_labels
            inputs["attention_mask"] = aug_attention_mask

        return inputs
    
    return collate_fn 
        

def inf_collate_fn(
    tokenizer,
    use_template: bool = True,
    max_length: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Callable[[List[Any]], Dict[str, torch.Tensor]]:

    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        template_kwargs = {}
        if _model_name_contains(model_name, "qwen3"):
            template_kwargs["enable_thinking"] = False
        inputs = tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            **template_kwargs,
        )
        return inputs

    return collate_fn

class ConversationDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def shuffle(self):
        import random
        random.shuffle(self.data_list)
        return self