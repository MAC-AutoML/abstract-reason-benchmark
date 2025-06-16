import json
import os
import time
import openai  
from tqdm import tqdm
import tiktoken  
from copy import deepcopy
import argparse
import re


parser = argparse.ArgumentParser(description='Test model on reasoning dataset.')
parser.add_argument('--model_name', type=str, default='deepseek-chat', help='Name of the OpenAI model to use.')
parser.add_argument('--key', type=str, help='OpenAI API key.')
parser.add_argument('--base_url', type=str, help='OpenAI API base URL.')
args = parser.parse_args()


openai.api_base = args.base_url
openai.api_key = args.key
model_name = args.model_name


print(f"Using model: {model_name}")
print(f"API base URL: {openai.api_base}")


tokenizer = tiktoken.get_encoding("cl100k_base")  

generation_config = {
    "temperature": 1e-7,  
    "max_new_tokens": 4096,  
}

def extract_answer(text):
    
    pattern = r"The answer is (.*?)(?:\.\s|\.\n|$)"
    match = re.search(pattern, text)
    if match:
        
        answer = match.group(1)
        
        answer = re.sub(r'[^\w\s\.,\{\}\[\]-]', '', answer)
        answer = answer.rstrip('.')
        return answer
    
    
    
    
    
    
    return ''

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  
    res = re.findall(r"(\d+(\.\d+)?)", text)  
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return None

def count_tokens(input_data, tokenizer):
    total_tokens = 0

    if isinstance(input_data, str):
        
        total_tokens = len(tokenizer.encode(input_data))
    elif isinstance(input_data, list):
        
        for message in input_data:
            if "content" in message:  
                content_tokens = tokenizer.encode(message["content"])
                total_tokens += len(content_tokens)
            else:
                print(f"Warning: Missing 'content' in message: {message}")
    else:
        raise TypeError("Input data must be a string or a list of messages.")

    return total_tokens


def generate_response(prompt, model_name=model_name, max_retries=60, delay=120):
    
    for p in prompt:
        p['type'] = 'text'
        if p['role'] == 'bot':
            p['role'] = 'assistant'
        if p['role'] == 'human':
            p['role'] = 'user'

    attempt = 0
    while attempt < max_retries:
        try:
            
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=prompt,
                max_tokens=generation_config["max_new_tokens"],
                temperature=generation_config["temperature"]
            )
            # print(response)
            # quit()
            
            return response.choices[0].message.content
        except openai.error.OpenAIError as e:
            
            print(f"Error occurred: {e}. Retrying... ({attempt + 1}/{max_retries})")
            attempt += 1
            if attempt < max_retries:
                time.sleep(delay)  
            else:
                raise  
        except Exception as e:
            
            print(f"Unexpected error occurred: {e}. Retrying... ({attempt + 1}/{max_retries})")
            attempt += 1
            if attempt < max_retries:
                time.sleep(delay)  
            else:
                raise  


def test_model_on_reason_dataset(file_path, level, cot=False, cot_type='zero'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]

    total_input_tokens, total_output_tokens = 0, 0
    output_records = []
    correct_count = 0

    for item in tqdm(data, desc=f"Processing {file_path}"):
        prompt = item["prompt"]
        answer = item["answer"]
        if cot:
            if cot_type == 'zero':
                    for j in range(len(prompt)):
                        if(prompt[j]['role'] == 'system'):
                            prompt[j]['content'] = "The task is to identify patterns and discover rules from the provided examples, then answer a question. The symbols in the question may not have their usual meanings, so carefully analyze the rules and expressions before providing your final answer in the format: 'Answer: The answer is {your answer}.'. Use step-by-step reasoning to explain your answer, even if examples only provide direct answers."
                    prompt[-1]['content'] =  prompt[-1]['content'] + "Let's think step by step. \n\n"
        
        generated_output = generate_response(prompt)
        
        
        
        input_tokens = count_tokens(prompt, tokenizer)
        output_tokens = count_tokens(generated_output, tokenizer)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        output = extract_answer(generated_output)

        
        if 'l0' in level or 'l3' in level:
            if (extract_last_num(output)) != None and extract_last_num(answer):
                is_correct = abs(extract_last_num(output) - extract_last_num(answer)) < 1e-2
            else:
                is_correct = 0
        else:
            is_correct = (answer == output)
            

        correct_count += is_correct

        
        output_records.append({
            "prompt": prompt,
            'raw_output': generated_output,
            "output": output,
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "is_correct": is_correct
        })

    accuracy = correct_count / len(data) if data else 0

    return accuracy, {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "output_records": output_records
    }


def run():

    cots = [False, True]
    cot_type = 'zero'
    levels = ['l0', 'l1', 'l2', 'l3', 'l4', 'l6']
    levels = ['l4']
    dataset_root_dir = 'dataset_1129'  
    results_root_dir = f'results_api_1129/{model_name}/result'  
    for cot in cots:
        if cot:
            results_root_dir += '_cot'
        os.makedirs(results_root_dir, exist_ok=True)

        
        total_input_tokens_all = 0
        total_output_tokens_all = 0

        for level in levels:
            level = level + '_chat'
            level_dir = os.path.join(dataset_root_dir, level)
            if not os.path.exists(level_dir):
                print(f"Level directory {level_dir} does not exist. Skipping.")
                continue

            datasets = [f for f in os.listdir(level_dir) if f.endswith('.json')]
            for dataset_file in datasets:
                if 'gsm8k_cot.json' in dataset_file and not cot:
                    continue
                if 'gsm8k.json' in dataset_file and cot:
                    continue
                dataset_path = os.path.join(level_dir, dataset_file)
                dataset_name = dataset_file.replace('.json', '')
                results_dir = os.path.join(results_root_dir, level, dataset_name)

                os.makedirs(results_dir, exist_ok=True)

                summary_file = os.path.join(results_dir, 'result.json')
                output_file = os.path.join(results_dir, 'output.json')
                print(summary_file)
                print(output_file)
                if os.path.exists(summary_file) and os.path.exists(output_file):
                    print(f"Results for {dataset_file} already exist. Skipping.")
                    continue

                
                start_time = time.time()
                accuracy, result_data = test_model_on_reason_dataset(dataset_path, level, cot=cot, cot_type=cot_type)
                end_time = time.time()

                
                total_input_tokens_all += result_data["total_input_tokens"]
                total_output_tokens_all += result_data["total_output_tokens"]

                
                result_summary = {
                    "accuracy": accuracy,
                    "token_usage": {"total_input_tokens":result_data["total_input_tokens"],
                                    "total_output_tokens": result_data["total_output_tokens"],
                                    'totle_tokens' : result_data["total_input_tokens"] + result_data["total_output_tokens"]},
                    "run_time_seconds": end_time - start_time
                }
                with open(summary_file, 'w') as f:
                    json.dump(result_summary, f, indent=4)

                
                with open(output_file, 'w') as f:
                    json.dump(result_data["output_records"], f, indent=4)

                print(f"Results saved to {summary_file}")
                print(f"Test outputs saved to {output_file}")

        
        total_tokens_summary = {
            "total_input_tokens": total_input_tokens_all,
            "total_output_tokens": total_output_tokens_all
        }

        
        total_tokens_file = os.path.join(results_root_dir, 'result.json')
        with open(total_tokens_file, 'w') as f:
            json.dump(total_tokens_summary, f, indent=4)

        print(f"Total token usage saved to {total_tokens_file}")

if __name__ == "__main__":
    run()