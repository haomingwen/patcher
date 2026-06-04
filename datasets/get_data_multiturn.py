from datasets import load_dataset
import torch
import random

def get_custom_data(split='train', num_benign=100, num_harmful=100, benign_path="datasets/custom_benign.json", harmful_path="datasets/custom_harmful.json", benign_multiturn=False, harmful_multiturn=False):
    def create_evaluation_data_single(prompts, responses):
        system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        evaluation_data = []
        for i in range(len(prompts)):
            sys_prompt = {'role': 'system', 'content': system_prompt}
            user_prompt = {'role': 'user', 'content': f"{prompts[i]}"}
            assistant_prompt = {'role': 'assistant', 'content': f"{responses[i]}"}
            input_sample = []
            input_sample.append(sys_prompt)
            input_sample.append(user_prompt)
            input_sample.append(assistant_prompt)
            evaluation_data.append(input_sample)
        return evaluation_data
    
    def create_evaluation_data_multi(messages):
        system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        evaluation_data = []
        for msg in messages:
            new_msg = [{'role': 'system', 'content': system_prompt}]
            for turn in msg:
                new_msg.append({'role': turn['role'], 'content': turn['content']})
            evaluation_data.append(new_msg)
        return evaluation_data
    
    benign_dataset = load_dataset("json", data_files=benign_path, split=split)
    harmful_dataset = load_dataset("json", data_files=harmful_path, split=split)
    benign_dataset = benign_dataset.shuffle(seed=42)
    harmful_dataset = harmful_dataset.shuffle(seed=42)
    benign_prompts = benign_dataset[:num_benign]
    if benign_multiturn:
        messages = [benign_data["messages"] for benign_data in benign_dataset][:min(len(benign_dataset), num_benign)]
        benign_evaluation_data = create_evaluation_data_multi(messages)
    else:
        benign_prompts = [benign_data["prompt"] for benign_data in benign_dataset][:num_benign]
        benign_responses = [benign_data["response"] for benign_data in benign_dataset][:num_benign]
        benign_evaluation_data = create_evaluation_data_single(benign_prompts, benign_responses)

    if harmful_multiturn:
        messages = [harmful_data["messages"] for harmful_data in harmful_dataset][:min(len(harmful_dataset), num_harmful)]
        harmful_evaluation_data = create_evaluation_data_multi(messages)
    else:
        harmful_prompts = [harmful_data["prompt"] for harmful_data in harmful_dataset][:num_harmful]
        harmful_responses = [harmful_data["response"] for harmful_data in harmful_dataset][:num_harmful]
        harmful_evaluation_data = create_evaluation_data_single(harmful_prompts, harmful_responses)
        
    evaluation_data = benign_evaluation_data + harmful_evaluation_data
    random.Random(42).shuffle(evaluation_data)

    return evaluation_data
