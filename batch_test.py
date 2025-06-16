import json
import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers

import time  
import random
import numpy as np
from transformers import set_seed
import re
import copy
from difflib import SequenceMatcher


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)


parser = argparse.ArgumentParser(description="Test a language model on a dataset")
parser.add_argument('--model_name', type=str, required=True, help="The name of the model to load")
parser.add_argument('--chat', action='store_true', help="Is chat model (default: False)")
parser.add_argument('--batch_size', type=int, default=2, help="Batch size for generation")
parser.add_argument('--cot', action='store_true', help="cot")
parser.add_argument('--cot_type', type=str, default='zero', help="The name of the model to load")
parser.add_argument('--dataset', type=str, default='dataset_1113', help="dataset")
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = args.model_name
batch_size = args.batch_size


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("load model: ", model_name)

generation_config = {
    "temperature": 1e-7,  
    "max_new_tokens": 4096,  
    "batch_size": batch_size  
}

import re

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

def is_semantically_similar(answer, output, threshold=0.9):
    ratio = SequenceMatcher(None, answer, output).ratio()
    return ratio >= threshold

def learning_gen(batch_prompts, generator, tokenizer):
    res = []
    total_input_tokens, total_output_tokens = 0, 0  
    choose_prompt = "Answer the question if possible using the format: 'The answer is {your answer}.' If you cannot answer, respond with 'Need more samples.'\n"
    final_prompt = "These are all the examples. Now you must answer this question, using the format: 'The answer is {your answer}.'\n"
    example_num = 0
    for prompt in batch_prompts:
        i = 0
        sys = [prompt[0]]
        q = prompt[-1]['content']
        
        
        prompt_copy = copy.deepcopy(prompt)
        prompt_copy[-1]['content'] = choose_prompt + q
        new_q = [prompt_copy[-1]]
        
        
        prompt[-1]['content'] = final_prompt + q
        final_q = [prompt[-1]]
        
        example = prompt[1:-1]
        new_prompt = sys + example[i:i + 10] + new_q

        while i < len(example):
            
            input_tokens = sum(count_tokens(p['content'], tokenizer) for p in new_prompt)
            total_input_tokens += input_tokens

            
            text_gen = generator(new_prompt)
            generated_text = text_gen[0]['generated_text'][-1]['content']

            
            output_tokens = count_tokens(generated_text, tokenizer)
            total_output_tokens += output_tokens

            
            if "more samples" not in generated_text:
                break
            else:
                i += 10
                example_num += len(example[i:i + 10])
                if(i + 10 >= len(example)):
                    new_prompt = text_gen[0]['generated_text'] + example[i:i + 10] + final_q
                else:
                    new_prompt = text_gen[0]['generated_text'] + example[i:i + 10] + new_q
        
        example_num += len(example[i:])
        res.append(text_gen)
        

    
    return res, total_input_tokens, total_output_tokens, example_num

def count_tokens(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def gen_qwen(model, tokenizer, prompts):
    
    texts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]
    
    
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=generation_config["max_new_tokens"],
        temperature=generation_config["temperature"]
    )
    
    
    input_lengths = [len(input_ids) for input_ids in model_inputs.input_ids]
    gen_text = [
        tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
        for output_ids, input_len in zip(generated_ids, input_lengths)
    ]
    
    return gen_text


def test_model_on_reason_dataset(file_path, level='l0', batch_size=1):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if 'l5' in level:
        batch_size = 1
    prompts = [item["prompt"] for item in data]
    answers = [item["answer"] for item in data]
    for prompt in prompts:
        for p in prompt:
            p['type'] = 'text'
            if(p['role'] == 'bot'):
                p['role'] = 'assistant'
            if(p['role'] == 'human'):
                p['role'] = 'user'
    if args.cot:
        if args.cot_type == 'zero':
            for i in range(len(prompts)):
                for j in range(len(prompts[i])):
                    
                    
                    
                    
                    if(prompts[i][j]['role'] == 'system'):
                        prompts[i][j]['content'] = "The task is to identify patterns and discover rules from the provided examples, then answer a question. The symbols in the question may not have their usual meanings, so carefully analyze the rules and expressions before providing your final answer in the format: 'Answer: The answer is {your answer}.'. Use step-by-step reasoning to explain your answer, even if examples only provide direct answers."
                prompts[i][-1]['content'] =  prompts[i][-1]['content'] + "Let's think step by step. \n\n"
    acc, cnt, total_input_tokens, total_output_tokens, total_example_num = 0, 0, 0, 0, 0

    if args.chat:
        generator = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=generation_config["max_new_tokens"],
            batch_size=generation_config["batch_size"],
            temperature=generation_config["temperature"]
        )
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {file_path}"):
            batch_prompts = prompts[i:i + batch_size]
            batch_answers = answers[i:i + batch_size]
            if 'l5' in level:
                gen_texts, input_tokens, output_tokens, example_num = learning_gen(batch_prompts, generator, tokenizer)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_example_num += example_num
            elif 'qwen' in model_name.lower():
                gen_texts = gen_qwen(model, tokenizer, batch_prompts)
            else:
                gen_texts = generator(batch_prompts)
                batch_prompts = [p[0]['generated_text'][:-1] for p in gen_texts] 
            
            
            
            for j, output in enumerate(gen_texts):
                if 'qwen' in model_name.lower():
                    generated_text = output
                else:
                    generated_text = output[0]['generated_text'][-1]['content']
                output_text = extract_answer(generated_text)

                
                if 'l5' not in level:
                    input_tokens = sum(count_tokens(p['content'], tokenizer) for p in batch_prompts[j])
                    output_tokens = count_tokens(generated_text, tokenizer)
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                
                if extract_last_num(batch_answers[j]) is not None and ('l0' in level or 'l3' in level or 'l6' in level):
                    if (extract_last_num(output_text)) != None:
                        is_correct = abs(extract_last_num(output_text) - extract_last_num(batch_answers[j])) < 1e-2
                    else:
                        is_correct = 0
                else:
                    is_correct = (batch_answers[j] == output_text)
                    

                acc += is_correct
                cnt += 1
                output_records.append({
                    "prompt": batch_prompts[j],
                    'raw_output': generated_text,
                    "output": output_text,
                    "answer": batch_answers[j],
                    'correct': is_correct,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                })
                
                
                
                
                
                
                
                
                
                

    else:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {file_path}"):
            batch_prompts = prompts[i:i + batch_size]
            batch_answers = answers[i:i + batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=25, temperature=0)

            for j, output in enumerate(outputs):
                output_text = tokenizer.decode(output, skip_special_tokens=True)
                output_text = output_text[len(batch_prompts[j]):].split('\n')[0]

                
                input_tokens = count_tokens(batch_prompts[j], tokenizer)
                output_tokens = count_tokens(output_text, tokenizer)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens


                if extract_last_num(batch_answers[j]) != None:
                    is_correct = abs(extract_last_num(output_text) - extract_last_num(batch_answers[j])) < 1e-2
                else:
                    is_correct = batch_answers[j] in output_text

                acc += is_correct
                cnt += 1
                output_records.append({
                    "prompt": batch_prompts[j],
                    "output": output_text,
                    "answer": batch_answers[j],
                    'correct': is_correct,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_example_num': total_example_num
                })

    accuracy = acc / cnt if cnt > 0 else 0
    total_tokens = {"total_input_tokens": total_input_tokens, "total_output_tokens": total_output_tokens, 'total_example_num': total_example_num}
    return accuracy, total_tokens


levels = ['l0', 'l1', 'l2', 'l3', 'l4', 'l6']



for level in levels: 
    if args.chat:
        level = level + '_chat'
        results_dir = f'result_chat_{args.dataset}/{model_name}/result'
    else:
        results_dir = f'result_{args.dataset}/{model_name}/result'
    if args.cot:
        results_dir = results_dir + '_' + args.cot_type
    dataset_dir = args.dataset + f'/{level}/'
    datasets = sorted(os.listdir(dataset_dir))
    config_file = os.path.join(results_dir, 'config.json')
    os.makedirs(results_dir, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(generation_config, f, indent=4)
    print(dataset_dir)
    for filename in datasets:
        if 'dataset' in filename and filename.endswith('.json'):
            if 'gsm8k_cot.json' in filename and not args.cot:
                continue
            if 'gsm8k.json' in filename and args.cot:
                continue
            
            
            res_dir = os.path.join(results_dir, level)

            
            file_results_dir = os.path.join(res_dir, filename.replace('.json', ''))
            os.makedirs(file_results_dir, exist_ok=True)

            results_file = os.path.join(file_results_dir, 'result.json')

            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    previous_results = json.load(f)
                    print(f"Results for {filename} already exist:")
                    
                continue  

            output_records = []
            file_path = os.path.join(dataset_dir, filename)

            start_time = time.time()

            accuracy, total_tokens = test_model_on_reason_dataset(file_path, level=level, batch_size=batch_size)
            
            end_time = time.time()
            run_time = end_time - start_time  

            

            with open(results_file, 'w') as f:
                json.dump({
                    filename: accuracy,
                    "token_usage": total_tokens,
                    "run_time_seconds": run_time  
                }, f, indent=4)
            print("accuracy: ", accuracy)
            print("Token usage:", total_tokens)
            print("Run time (seconds):", run_time)
            
            
            output_file = os.path.join(file_results_dir, 'output.json')
            with open(output_file, 'w') as f:
                json.dump(output_records, f, indent=4)

            print(f"Results saved to {results_file}")
            print(f"Test outputs saved to {output_file}")