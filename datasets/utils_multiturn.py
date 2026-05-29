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


def _build_text(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        return _format_messages_with_template(tokenizer, messages, add_generation_prompt=False)
    return _format_messages_fallback(messages)


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

def _get_assistant_header_text(model_name: Optional[str]) -> str:
    if model_name == "qwen":
        return "<|im_start|>assistant\n"
    if model_name == "llama":
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return ""


def _get_headers_len(tokenizer, model_name: Optional[str] = None) -> int:
    """
    Get the token length of the assistant header only.
    """
    header_text = _get_assistant_header_text(model_name)
    if not header_text:
        return 0
    return len(tokenizer(header_text, add_special_tokens=False).input_ids)

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
                labels_mask = None
                if mask_prompts:
                    labels_mask = [1] * len(tokenizer(full_text, add_special_tokens=False).input_ids)
            else:
                full_text = _build_text(tokenizer, messages, use_template=use_template)
                labels_mask = None
                if mask_prompts:
                    labels_mask = []
                    cur_messages = []
                    prev_messages = []
                    prev_extra_len = _get_headers_len(tokenizer, model_name=model_name)
                    for msg in messages:
                        cur_messages.append(msg)
                        cur_text = _build_text(tokenizer, cur_messages, use_template=use_template)
                        if prev_messages:
                            prev_text = _build_text(tokenizer, prev_messages, use_template=use_template)
                        else:
                            prev_text = ""
                        prev_len = len(tokenizer(prev_text, add_special_tokens=False).input_ids)
                        cur_len = len(tokenizer(cur_text, add_special_tokens=False).input_ids)
                        if msg.get("role") == "assistant":
                            # account for headers
                            token_span = cur_len - prev_len - prev_extra_len
                            labels_mask.extend([0] * prev_extra_len)
                            labels_mask.extend([1] * token_span)
                        else:
                            token_span = cur_len - prev_len
                            labels_mask.extend([0] * token_span)
                        prev_messages.append(msg)

            encoded = tokenizer(full_text, add_special_tokens=False)
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            if mask_prompts:
                if labels_mask is None or len(labels_mask) != labels.shape[0]:
                    raise ValueError(
                        "labels_mask length does not match tokenized input length: "
                        f"{0 if labels_mask is None else len(labels_mask)} vs {labels.shape[0]}"
                    )
                labels_mask_tensor = torch.tensor(labels_mask, dtype=torch.bool)
                labels = torch.where(labels_mask_tensor, labels, torch.full_like(labels, -100))

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
