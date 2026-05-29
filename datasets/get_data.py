from datasets import load_dataset
import torch
import random
def get_repnoise(split='train'):

    def process_refusal(text):
        # only retain the refusal text after "Answer: "
        if "Answer: " in text:
            return text.split("Answer: ")[-1].strip()
        else:
            raise ValueError("Refusal text does not contain 'Answer: '")
    
    def split_safe_responses_pairs(dataset):
        safe_prompts = []
        safe_responses = []
        unsafe_prompts = []
        unsafe_responses = []
        for item in dataset:
            if not item['is_safe']:
                unsafe_prompts.append(item['prompt'])
                unsafe_responses.append(item['response'])
                safe_prompts.append(item['prompt'])
                safe_responses.append(process_refusal(item['refusal']))
        
        return safe_prompts, safe_responses, unsafe_prompts, unsafe_responses

    def create_evaluation_data(prompts, responses):
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
    
    dataset = load_dataset("json", data_files=f"datasets/beavertails_with_refusals_train.json", split='train')
    safe_prompts, safe_responses, unsafe_prompts, unsafe_responses = split_safe_responses_pairs(dataset)
    safe_evaluation_data = create_evaluation_data(safe_prompts, safe_responses)
    unsafe_evaluation_data = create_evaluation_data(unsafe_prompts, unsafe_responses)

    return safe_evaluation_data, unsafe_evaluation_data

def get_custom(split='train', num_benign=100, num_harmful=100, benign_path="datasets/custom_benign.json", harmful_path="datasets/custom_harmful.json"):
    def create_evaluation_data(prompts, responses):
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
    
    benign_dataset = load_dataset("json", data_files=benign_path, split=split)
    harmful_dataset = load_dataset("json", data_files=harmful_path, split=split)
    benign_dataset = benign_dataset.shuffle(seed=42)
    harmful_dataset = harmful_dataset.shuffle(seed=42)
    
    benign_prompts = [benign_data["prompt"] for benign_data in benign_dataset]
    if num_benign is not None:
        benign_prompts = benign_prompts[:min(num_benign, len(benign_prompts))]

    benign_responses = [benign_data["response"] for benign_data in benign_dataset]
    if num_benign is not None:
        benign_responses = benign_responses[:min(num_benign, len(benign_responses))]

    harmful_prompts = [harmful_data["prompt"] for harmful_data in harmful_dataset]
    if num_harmful is not None:
        harmful_prompts = harmful_prompts[:min(num_harmful, len(harmful_prompts))]

    harmful_responses = [harmful_data["response"] for harmful_data in harmful_dataset]
    if num_harmful is not None:
        harmful_responses = harmful_responses[:min(num_harmful, len(harmful_responses))]

    benign_evaluation_data = create_evaluation_data(benign_prompts, benign_responses)
    harmful_evaluation_data = create_evaluation_data(harmful_prompts, harmful_responses)
    evaluation_data = benign_evaluation_data + harmful_evaluation_data
    random.Random(42).shuffle(evaluation_data)

    return evaluation_data


def get_repnoise_augment(split='train'):

    def process_refusal(text):
        # only retain the refusal text after "Answer: "
        if "Answer: " in text:
            return text.split("Answer: ")[-1].strip()
        else:
            raise ValueError("Refusal text does not contain 'Answer: '")
    
    def split_safe_responses_pairs(dataset):
        safe_prompts = []
        safe_responses = []
        unsafe_prompts = []
        unsafe_responses = []
        for item in dataset:
            if not item['is_safe']:
                unsafe_prompts.append(item['prompt'])
                unsafe_responses.append(item['response'])
                safe_prompts.append(item['prompt'])
                safe_responses.append(process_refusal(item['refusal']))
        
        return safe_prompts, safe_responses, unsafe_prompts, unsafe_responses

    def create_evaluation_data(prompts, responses, prefixes=None):
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
            if prefixes is not None:
                assistant_adv = {'role': 'assistant', 'content': f"{prefixes[i]}"}
                input_sample.append(assistant_adv)
            evaluation_data.append(input_sample)
        return evaluation_data
    
    dataset = load_dataset("json", data_files=f"datasets/beavertails_with_refusals_train.json", split='train')
    safe_prompts, safe_responses, unsafe_prompts, unsafe_responses = split_safe_responses_pairs(dataset)
    safe_evaluation_data = create_evaluation_data(safe_prompts, safe_responses, unsafe_responses)
    unsafe_evaluation_data = create_evaluation_data(unsafe_prompts, unsafe_responses)

    return safe_evaluation_data, unsafe_evaluation_data

def get_repnoise_eval(split='train'):
    
    def split_safe_responses_pairs(dataset):
        prompts = []
        for item in dataset:
            if not item['is_safe']:
                prompts.append(item['prompt'])
        
        return prompts

    def create_evaluation_data(prompts):
        system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        evaluation_data = []
        for i in range(len(prompts)):
            sys_prompt = {'role': 'system', 'content': system_prompt}
            user_prompt = {'role': 'user', 'content': f"{prompts[i]}"}
            input_sample = []
            input_sample.append(sys_prompt)
            input_sample.append(user_prompt)
            evaluation_data.append(input_sample)
        return evaluation_data
    
    dataset = load_dataset("json", data_files=f"datasets/beavertails_with_refusals_train.json", split='train')
    prompts = split_safe_responses_pairs(dataset)
    evaluation_data = create_evaluation_data(prompts)

    return evaluation_data

def select_unsafe_beavertails(split='train'):

    def split_safe_responses_pairs(dataset):
        unsafe_data = []
        for item in dataset:
            if not item['is_safe']:
                unsafe_data.append(
                    {
                        "prompt": item['prompt'],
                        "response": item['response']
                    }
                )    
        
        return unsafe_data

    dataset = load_dataset("json", data_files=f"datasets/beavertails_train.json", split='train')
    unsafe_data = split_safe_responses_pairs(dataset)

    return unsafe_data

def get_beavertails(split='train'):

    def split_safe_responses_pairs(dataset):
        safe_prompts = []
        safe_responses = []
        unsafe_prompts = []
        unsafe_responses = []
        for item in dataset:
            # only use response 0
            if not item['is_safe']:
                unsafe_prompts.append(item['prompt'])
                unsafe_responses.append(item['response'])
            else:
                safe_prompts.append(item['prompt'])
                safe_responses.append(item['response'])
        return safe_prompts, safe_responses, unsafe_prompts, unsafe_responses
    
    def create_evaluation_data(prompts, responses):
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

    dataset = load_dataset("json", data_files=f"datasets/beavertails_330k.json", split='train')

    safe_prompts, safe_responses, unsafe_prompts, unsafe_responses = split_safe_responses_pairs(dataset)
    
    safe_evaluation_data = create_evaluation_data(safe_prompts, safe_responses)
    unsafe_evaluation_data = create_evaluation_data(unsafe_prompts, unsafe_responses)

    return safe_evaluation_data, unsafe_evaluation_data

def get_eval_for_generation(dataset_name, split='train'):
    
    dataset = load_dataset("json", data_files=f"datasets/{dataset_name}.json", split=split)

    system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    evaluation_data = []
    for item in dataset:
        sys_prompt = {'role': 'system', 'content': system_prompt}
        user_prompt = {'role': 'user', 'content': item['prompt']}
        input_sample = []
        input_sample.append(sys_prompt)
        input_sample.append(user_prompt)
        evaluation_data.append(input_sample)
    
    return evaluation_data
    

def get_alpaca(split='train'):

    def create_evaluation_data(data):
        evaluation_data = []
        for sample in data:
            sys_prompt = {'role': 'system', 'content': "Below is an instruction that describes a task. Write a response that appropriately completes the request."}
            user_text = f"###Instruction: {sample['instruction']}\n###Input: {sample['input']}"
            user_prompt = {'role': 'user', 'content': user_text}
            assistant_prompt = {'role': 'assistant', 'content': sample["output"]}
            input_sample = []
            input_sample.append(sys_prompt)
            input_sample.append(user_prompt)
            input_sample.append(assistant_prompt)
            evaluation_data.append(input_sample)
        return evaluation_data
        
    dataset = load_dataset("json", data_files="datasets/alpaca.json", split=split)
    evaluation_data = create_evaluation_data(dataset)
    return evaluation_data

def select_gsm8k(split='train'):

    def tranform(data):
        evaluation_data = []
        for sample in data:
            evaluation_data.append(
                {
                    "prompt": sample["question"],
                    "response": sample["answer"]
                }
            )
        return evaluation_data
    dataset = load_dataset("json", data_files="datasets/gsm8k.json", split=split)
    evaluation_data = tranform(dataset)
    return evaluation_data

