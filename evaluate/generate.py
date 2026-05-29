import torch
from tqdm import tqdm
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_harmfulness(
    model,
    eval_dataloader,
    tokenizer,
    dataset,
    use_sampler=False
) -> list:
    model.eval()
    generated_texts = []
    generated_texts_by_category = {}
    for batch in tqdm(eval_dataloader):
        max_new_tokens = 256
        if dataset == 'decoding_trust':
            max_new_tokens = 25
        params = {
            "max_new_tokens": max_new_tokens,
        }
        if use_sampler:
            params = {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.2,
            }
        with torch.inference_mode():
            try:
                outputs = model.generate(
                    batch['input_ids'].to(model.device),
                    attention_mask=batch['attention_mask'].to(model.device),
                    **params,
                )
            except:
                outputs = model.generate(
                    input_ids=batch['input_ids'].to(model.device),
                    attention_mask=batch['attention_mask'].to(model.device),
                    **params,
                )
        for i, output in enumerate(outputs):
            predicted_tokens = tokenizer.decode(
                output[len(batch['input_ids'][i]):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            if 'category' in batch:
                if batch['category'][i] not in generated_texts_by_category:
                    generated_texts_by_category[batch['category'][i]] = []        
                generated_texts_by_category[batch['category'][i]].append(predicted_tokens)
            else:
                generated_texts.append(predicted_tokens)
    return generated_texts
