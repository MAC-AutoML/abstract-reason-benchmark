import re
import json
from transformers import LlamaTokenizer
import os
import shutil
from rule import *
import itertools
import string
import copy
import random
import numpy as np
import torch
from transformers import set_seed
seed = 42 
random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


tokens = tokenizer.get_vocab()


pattern = re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9，。！？；：“”‘’《》、\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7A3\u00C0-\u00FF]+$')
pattern = re.compile(r'^[a-zA-Z0-9]+$')

length_one_non_standard_tokens = {
    token: idx for token, idx in tokens.items()
    if len(token) == 1 
    
    and pattern.match(token) 

    and token not in {' ', '\n', '\r'} 
    
}


output_dir = "dataset"
for i in range(7):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+f'/l{i}', exist_ok=True)
    os.makedirs(output_dir+f'/l{i}_chat', exist_ok=True)


output_file_path = os.path.join(output_dir, "tokens.json")
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(length_one_non_standard_tokens, f, ensure_ascii=False, indent=4)

print('-' * 30)
print(f"Tokens save to {output_file_path}")

max_val_list = 9

import random

def copy_files(source_dir, target_dir, without_list=['_op_', '_num_']):
    
    os.makedirs(target_dir, exist_ok=True)

    
    for filename in os.listdir(source_dir):
        if not any(substring in filename for substring in without_list):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            if os.path.isfile(source_file):  
                shutil.copy(source_file, target_file)
                


def random_token(tokens):

    if not tokens:
        return None  

    token = random.choice(list(tokens.keys()))  
    return token

def generate_random_integers(num, n):
    result = set()  
    num = min(num, power(10, n) - 1)
    while len(result) < num:
        
        num_digits = random.randint(1, n)
        
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        result.add(random.randint(lower_bound, upper_bound))
    return list(result)

def generate_random_floats(num, lower_bound, upper_bound):
    result = set()  
    while len(result) < num:
        result.add(random.uniform(lower_bound, upper_bound))
    return list(result)

def generate_random_bit_strings(num, length_bit):
    num = min(power(2, length_bit)-1, num)
    result = set()  
    while len(result) < num:
        bit_string = format(random.randint(128, 255), f'0{length_bit}b')
        result.add(bit_string)
    return list(result)

def generate_sequential_bit_strings(num, length_bit):
    num = min(power(2, length_bit)-1, num)
    return [format(i, f'0{length_bit}b') for i in range(num)]

def generate_random_sets(num, length_set):
    result = set()  
    num = min(power(2, length_set)-1, num)
    while len(result) < num:
        result.add(frozenset(random.randint(0, 10) for _ in range(length_set)))
    return [set(s) for s in result]

def generate_all_sets():
    
    elements = list(range(7))  
    all_sets = []
    
    
    for r in range(len(elements) + 1):  
        combinations = itertools.combinations(elements, r)
        all_sets.extend(combinations)
    
    
    return [set(combination) for combination in all_sets]

def generate_lists(n, m, min_val, max_val):
    result = set()  
    n = min(power(2, max_val - min_val)-1, n)
    while len(result) < n:
        result.add(tuple(random.randint(min_val, max_val) for _ in range(m)))
    return [list(t) for t in result]

def generate_long_word(length, letter_num=26):
    
    letters = string.ascii_lowercase[:letter_num]
    
    word = ''.join(random.choice(letters) for _ in range(length))
    
    ch = random.choice(word)
    
    return word, ch

def generate_words(num, length):
    words = set()  
    num_a = num // 2
    num_b = num - num_a
    while len(words) < num_a:
        words.add(generate_long_word(length, 26))
    while len(words) < num:
        words.add(generate_long_word(length, 8))
    return list(words)

def get_qa(name, data, char, result):
    if(len(data) > 1):
        return [(' ' + char + ' ').join(map(str, data)) + ' =', str(result)]
    elif(len(data) <= 1):
        return [char + ' ' + str(data[0]) + ' =', str(result)]
    


def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def get_example(ops, datas, test=True, type='bit', length='fixed', base=3):
    res = []
    if test: 
        data_len = 1
    else:
        data_len = 51

    if type == 'bit':
        for char in ops.keys():
            
            if length == 'fixed':
                l = len(datas[0])
            else:
                l = random.randint(1, len(datas[0]))

            
            sampled_data = random.sample(datas, data_len * ops[char]['args'])

            
            data_samples = chunk_list(sampled_data, ops[char]['args'])  
            
            
            for i, data in enumerate(data_samples):
                data.append(l)  
                fun = ops[char]['fun']
                name = ops[char]['name']
                res.append(get_qa(name, data[:-1], char, fun(*data)))

    elif type == 'bit_op':
        for char in ops.keys():
            
            if length == 'fixed':
                l = len(datas[0])
            else:
                l = random.randint(1, len(datas[0]))

            
            sampled_data = random.sample(datas, data_len * ops[char]['args'])

            
            data_samples = chunk_list(sampled_data, ops[char]['args'])

            
            for i, data in enumerate(data_samples):
                data = [d[:l] for d in data] + random.sample(range(l), ops[char]['parameter'])
                data.append(l)  
                fun = ops[char]['fun']
                name = ops[char]['name']
                res.append(get_qa(name, data[:-1], char, fun(*data)))


    elif type == 'set':
        for char in ops.keys():
            
            sampled_data = random.sample(datas, data_len * ops[char]['args'])

            
            data_samples = chunk_list(sampled_data, ops[char]['args'])

            
            for i, data in enumerate(data_samples):
                fun = ops[char]['fun']
                name = ops[char]['name']
                res.append(get_qa(name, data, char, fun(*data)))


    elif type == 'str':
        for char in ops.keys():
            
            if length == 'fixed':
                l = len(datas[0][0])
            else:
                l = random.randint(1, len(datas[0][0]))

            
            sampled_data = random.sample(datas, data_len * ops[char]['args'])
            sampled_data = [list(d) for d in sampled_data]
            data_samples = chunk_list(sampled_data, ops[char]['args'])
            
            
            for i, data in enumerate(data_samples):
                data = data[0]
                data[0] = data[0][:l]
                fun = ops[char]['fun']
                name = ops[char]['name']
                res.append(get_qa(name, data, char, fun(*data)))


    elif type == 'l0':
        for char in ops.keys():
            
            sampled_data = random.sample(datas, data_len * ops[char]['args'])

            
            data_samples = chunk_list(sampled_data, ops[char]['args'])

            
            for i, data in enumerate(data_samples):
                fun = ops[char]['fun']
                name = ops[char]['name']
                res.append(get_qa(name, data, char, fun(*data)))


    elif type == 'base':
        for char in ops.keys():
            
            sampled_data = random.sample(datas, data_len * ops[char]['args'])

            
            data_samples = chunk_list(sampled_data, ops[char]['args'])

            
            for i, data in enumerate(data_samples):
                data = [base] + data
                fun = ops[char]['fun']
                name = ops[char]['name']
                if name == 'to base':
                    res.append(get_qa(name, data[::-1], char, fun(*data)))
                else:
                    res.append(get_qa(name, data[1:], char, fun(*data)))


    elif type == 'linear':
        for char in ops.keys():
            num_param = ops[char]['parameter']
            
            
            sampled_data = random.sample(datas[0], data_len * ops[char]['args'])
            sampled_param = [random.sample(datas[1], num_param) for _ in range(data_len)]
            
            data_samples = chunk_list(sampled_data, ops[char]['args'])

            
            for i, data in enumerate(data_samples):
                data += sampled_param[i]
                fun = ops[char]['fun']
                name = ops[char]['name']
                res.append(get_qa(name, data[:-num_param], char, fun(*data)))

    elif type == 'list':
        for char in ops.keys():
            num_param = ops[char]['parameter']
            
            sampled_data = random.sample(datas, data_len * ops[char]['args'])
            sampled_param = [random.sample(range(max_val_list), num_param) for _ in range(data_len)]
            
            data_samples = chunk_list(sampled_data, ops[char]['args'])

            
            for i, data in enumerate(data_samples):
                data += sampled_param[i]
                fun = ops[char]['fun']
                name = ops[char]['name']
                res.append(get_qa(name, data, char, fun(*data)))


    elif type == 'substr':
        for char in ops.keys():
            strs, substrs = datas[0], datas[1]
            
            
            sampled_strs = random.sample(strs, data_len)
            sampled_substrs = random.sample(substrs, data_len)

            
            data_samples = [[s, sub] for s, sub in zip(sampled_strs, sampled_substrs)]

            
            for i, data in enumerate(data_samples):
                fun = ops[char]['fun']
                name = ops[char]['name']
                res.append(get_qa(name, data, char, fun(*data)))


    return res

def generate_random_base(n, base_n, length):
    num = '0123456789abcd'
    result = set()  
    n = min(n, power(10, length)-1)
    while len(result) < n:
        random_length = random.randint(1, length)  
        random_num = ''.join(random.choice(num[:base_n]) for _ in range(random_length))
        while random_num[0] == '0' and len(random_num) > 1:
            random_num = random_num[1:]  
        result.add(random_num)
    return list(result)

def generate_date_list(n, begin_year, end_year):

    date_list = set()  
    while len(date_list) < n:
        
        year = random.randint(begin_year, end_year)
        month = random.randint(1, 12)
        
        
        if month == 2:  
            day = random.randint(1, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28)
        elif month in {4, 6, 9, 11}:  
            day = random.randint(1, 30)
        else:  
            day = random.randint(1, 31)

        
        date_list.add((year, month, day))
        
    return [list(date) for date in date_list]



def unique_characters(question_ans, example):
    
    unique_chars = set()
    for string1, string2 in question_ans:
        
        unique_chars.update(char for char in string1 if char not in [' ', '\n'])
        unique_chars.update(char for char in string2 if char not in [' ', '\n'])
    for string1, string2 in example:
        
        unique_chars.update(char for char in string1 if char not in [' ', '\n'])
        unique_chars.update(char for char in string2 if char not in [' ', '\n'])
    
    return unique_chars

def get_random_token_map(chs):
    chs = list(chs)
    random_token_char_idx = random.sample(range(len(special_tokens)), len(chs))
    random_token_char = [special_tokens[idx] for idx in random_token_char_idx]
        
    ch2token = dict(zip(chs, random_token_char))
    token2ch = dict(zip(random_token_char, chs))

    return ch2token, token2ch

def map_string_chars(datas, char_map):
    
    map_data = copy.deepcopy(datas)
    for i in range(len(datas)):
        for j in range(len(datas[i])):
            map_data[i][j] = ''.join([char_map[char] if char in char_map else char for char in datas[i][j]])
    return map_data



def get_prompt(example, qa, chat=False, hint=''):
    if chat:
        prompt = [{
                    "role": "system",
                    "content": base + hint
                    }]
        for eq, ea in example:
            prompt.append({
                        "role": "human",
                        "content": hint + "Question: " + eq + '\n\n'
                    })
            prompt.append({
                        "role": "bot",
                        "content": "Answer: The answer is " + ea + '.\n\n'
                    })
        prompt.append({
                        "role": "human",
                        "content": hint + "Question: " + qa[0][0] + '\n\n'
                    })
        return prompt
    else:
        prompt = '' + hint
        for eq, ea in example:
            prompt = prompt + f'{eq}{ea}\n'
        prompt = prompt + f'{qa[0][0]}'
        return prompt

if __name__ == "__main__":

    num_op = len(bit_function_list)
    
    special_tokens = list(length_one_non_standard_tokens.keys())

    print("here are some special tokens : ", special_tokens[:30])

    bit_length = 8
    example_data = range(1, 6)

    
    num_data = 96
    random_integers = generate_random_integers(1000, 9)
    random_integers_mid = generate_random_integers(1000, 4)
    random_integers_small = generate_random_integers(1000, 2)
    param_intsers_small = generate_random_integers(1000, 1)
    print('random_integers: ', random_integers[:10])
    
    
    

    
    random_bitstr = generate_random_bit_strings(1, bit_length)
    
    seq_bitstr = generate_sequential_bit_strings(128, bit_length)
    
    all_sets = generate_all_sets()
    random_words = generate_words(1000, 40)
    random_strs = [str for (str, ch) in random_words]
    random_words_small = generate_words(1000, 4)
    random_strs_small = [str for (str, ch) in random_words_small]
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    base_random_bitstr = [generate_random_base(1000, 3, 8), 
                          generate_random_base(1000, 4, 8),
                          generate_random_base(1000, 5, 8)]
    print(f"Generated random_words: {random_words[:10]}")
    print('random_set: ', all_sets[:10])
    print('random_bitstr: ', random_bitstr[:10])
    print('seq_bitstr: ', seq_bitstr)
    print(bit_function_list)
    
    base = "The task is to identify patterns and discover rules from the provided examples, then complete the following expressions. The symbols in the question may not have their usual meanings, so carefully analyze the rules and expressions before providing your final answer in the format: 'Answer: your answer.'."
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []

    
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(bit_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [bit_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, seq_bitstr, test=False, length='fixed')
        example, question_ans= example[:-1], example[-1:]
        
        


        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})

    
    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bit_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1/fixed_len_bit_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bit_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/fixed_len_chat_bit_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bit_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bit_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bit_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bit_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)


    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(bit_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [bit_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, seq_bitstr, test=False, length='var')
        example, question_ans= example[:-1], example[-1:]
        
        
        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        


    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bit_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_bit_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
    
    
    dataset_file_path = os.path.join(output_dir, "l4/var_len_bit_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1/var_len_bit_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_bit_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bit_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_bit_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bit_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []

    
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(bit_shift_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [bit_shift_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, seq_bitstr, test=False, length='fixed', type='bit_op')
        example, question_ans= example[:-1], example[-1:]
        
        


        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})

    
    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bit_shift_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1/fixed_len_bit_shift_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bit_shift_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/fixed_len_chat_bit_shift_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bit_shift_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bit_shift_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bit_shift_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bit_shift_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []

    
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(bit_shift_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [bit_shift_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, seq_bitstr, test=False, length='var', type='bit_op')
        example, question_ans= example[:-1], example[-1:]
        
        


        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})

    
    dataset_file_path = os.path.join(output_dir, "l4/var_len_bit_shift_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1/var_len_bit_shift_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bit_shift_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_bit_shift_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_bit_shift_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bit_shift_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_bit_shift_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bit_shift_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []

    
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(bit_op_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [bit_op_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, seq_bitstr, test=False, length='var', type='bit_op')
        example, question_ans= example[:-1], example[-1:]
        
        


        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})

    
    dataset_file_path = os.path.join(output_dir, "l4/var_len_bitop_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1/var_len_bitop_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bitop_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_bitop_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_bitop_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bitop_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_bitop_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_bitop_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []

    
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(bit_op_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [bit_op_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, seq_bitstr, test=False, length='fixed', type='bit_op')
        example, question_ans= example[:-1], example[-1:]
        
        


        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})

    
    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bitop_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1/fixed_len_bitop_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bitop_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/fixed_len_chat_bitop_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bitop_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bitop_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_bitop_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_bitop_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(set_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [set_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, all_sets, test=False, type='set')
        example, question_ans= example[:-1], example[-1:]
        

        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_set_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_set_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_set_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/var_len_set_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l4/var_len_set_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_set_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_set_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_set_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(str_function_list)), num_random_op)
        selected_functions = [str_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, random_words, test=False, type='str', length='var')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_str_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_str_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l4/var_len_str_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/var_len_str_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_str_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_str_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_str_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_str_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(str_function_list)), num_random_op)
        selected_functions = [str_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, random_words, test=False, type='str', length='fixed')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    
    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_str_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/fixed_len_chat_str_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
        
    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_str_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/fixed_len_str_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_str_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_str_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_str_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_str_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(str_op_function_list)), num_random_op)
        selected_functions = [str_op_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, random_strs, test=False, type='set', length='fixed')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_strop_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/fixed_len_chat_strop_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
        
    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_strop_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/fixed_len_strop_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_strop_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_strop_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_strop_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_strop_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(str_op_function_list)), num_random_op)
        selected_functions = [str_op_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, random_strs, test=False, type='set', length='var')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_strop_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_strop_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
        
    dataset_file_path = os.path.join(output_dir, "l4/var_len_strop_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/var_len_strop_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_strop_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_strop_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_strop_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_strop_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)
    
    os.makedirs(output_dir, exist_ok=True)
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(add_function_list)), num_random_op)
        selected_functions = [add_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))
        
        
        example = get_example(ops_dict, random_integers, test=False, type='l0', length='var')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)

        
        
        

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        
        

        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        
        
        

    
    
    dataset_file_path = os.path.join(output_dir, "l0_chat/chat_add_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
        
    
    dataset_file_path = os.path.join(output_dir, "l0/add_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(sub_function_list)), num_random_op)
        selected_functions = [sub_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))
        
        
        example = get_example(ops_dict, random_integers, test=False, type='l0', length='var')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)

        
        
        

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        
        

        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        
        
        

    
    
    dataset_file_path = os.path.join(output_dir, "l0_chat/chat_sub_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
        
    
    dataset_file_path = os.path.join(output_dir, "l0/sub_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(mul_function_list)), num_random_op)
        selected_functions = [mul_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))
        
        
        example = get_example(ops_dict, random_integers, test=False, type='l0', length='var')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)

        
        
        

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        
        

        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        
        
        

    
    
    dataset_file_path = os.path.join(output_dir, "l0_chat/chat_mul_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
        
    
    dataset_file_path = os.path.join(output_dir, "l0/mul_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(div_function_list)), num_random_op)
        selected_functions = [div_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))
        
        
        example = get_example(ops_dict, random_integers, test=False, type='l0', length='var')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)

        
        
        

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        
        

        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        
        
        

    
    
    dataset_file_path = os.path.join(output_dir, "l0_chat/chat_div_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
        
    
    dataset_file_path = os.path.join(output_dir, "l0/div_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(square_function_list)), num_random_op)
        selected_functions = [square_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))
        
        
        example = get_example(ops_dict, random_integers, test=False, type='l0', length='var')
        example, question_ans= example[:-1], example[-1:]
        
        unique_ch = unique_characters(question_ans, example)

        
        
        

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        
        

        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        
        
        

    
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/chat_square_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
        
    
    dataset_file_path = os.path.join(output_dir, "l1/square_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)
    
    def base_data_gen(base=3):
        hint = f'\nHint: This is base {base} operation.\n'
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        raw_chat_dataset = []
        raw_dataset = []
        chat_dataset = []
        dataset = []
        for i in range(num_data):
            num_random_op = 1
            random_op_idx = random.sample(range(len(add_base_function_list)), num_random_op)
            selected_functions = [add_base_function_list[idx] for idx in random_op_idx]
            random_op_char =  [selected_function['name'] for selected_function in selected_functions]
            ops_dict = dict(zip(random_op_char, selected_functions))
            
            
            example = get_example(ops_dict, base_random_bitstr[base-3], test=False, type='base', length='var', base=base)
            example, question_ans= example[:-1], example[-1:]
            
            unique_ch = unique_characters(question_ans, example)

            ch2token, token2ch = get_random_token_map(unique_ch)
            new_example = map_string_chars(example, ch2token)
            new_question_ans = map_string_chars(question_ans, ch2token)

            raw_chat_prompt = get_prompt(example, question_ans, chat=True, hint=hint)
            raw_prompt = get_prompt(example, question_ans, chat=False, hint=hint)
            chat_prompt = get_prompt(new_example, new_question_ans, chat=True, hint=hint)
            prompt = get_prompt(new_example, new_question_ans, chat=False, hint=hint)

            raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
            raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
            chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
            dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
            

        
        
        dataset_file_path = os.path.join(output_dir, f"l2_chat/chat_add_base{base}_raw_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
            
        
        dataset_file_path = os.path.join(output_dir, f"l2/add_base{base}_raw_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

        
        
        
            
        
        
        
        
        
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        raw_chat_dataset = []
        raw_dataset = []
        chat_dataset = []
        dataset = []
        for i in range(num_data):
            num_random_op = 1
            random_op_idx = random.sample(range(len(sub_base_function_list)), num_random_op)
            selected_functions = [sub_base_function_list[idx] for idx in random_op_idx]
            random_op_char =  [selected_function['name'] for selected_function in selected_functions]
            ops_dict = dict(zip(random_op_char, selected_functions))
            
            
            example = get_example(ops_dict, base_random_bitstr[base-3], test=False, type='base', length='var', base=base)
            example, question_ans= example[:-1], example[-1:]
            
            unique_ch = unique_characters(question_ans, example)

            ch2token, token2ch = get_random_token_map(unique_ch)
            new_example = map_string_chars(example, ch2token)
            new_question_ans = map_string_chars(question_ans, ch2token)

            raw_chat_prompt = get_prompt(example, question_ans, chat=True, hint=hint)
            raw_prompt = get_prompt(example, question_ans, chat=False, hint=hint)
            chat_prompt = get_prompt(new_example, new_question_ans, chat=True, hint=hint)
            prompt = get_prompt(new_example, new_question_ans, chat=False, hint=hint)

            raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
            raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
            chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
            dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
            

        
        
        dataset_file_path = os.path.join(output_dir, f"l2_chat/chat_sub_base{base}_raw_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
            
        
        dataset_file_path = os.path.join(output_dir, f"l2/sub_base{base}_raw_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

        
        
        
            
        
        
        
        
        
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        raw_chat_dataset = []
        raw_dataset = []
        chat_dataset = []
        dataset = []
        for i in range(num_data):
            num_random_op = 1
            random_op_idx = random.sample(range(len(mul_base_function_list)), num_random_op)
            selected_functions = [mul_base_function_list[idx] for idx in random_op_idx]
            random_op_char =  [selected_function['name'] for selected_function in selected_functions]
            ops_dict = dict(zip(random_op_char, selected_functions))
            
            
            example = get_example(ops_dict, base_random_bitstr[base-3], test=False, type='base', length='var', base=base)
            example, question_ans= example[:-1], example[-1:]
            
            unique_ch = unique_characters(question_ans, example)

            ch2token, token2ch = get_random_token_map(unique_ch)
            new_example = map_string_chars(example, ch2token)
            new_question_ans = map_string_chars(question_ans, ch2token)

            raw_chat_prompt = get_prompt(example, question_ans, chat=True, hint=hint)
            raw_prompt = get_prompt(example, question_ans, chat=False, hint=hint)
            chat_prompt = get_prompt(new_example, new_question_ans, chat=True, hint=hint)
            prompt = get_prompt(new_example, new_question_ans, chat=False, hint=hint)

            raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
            raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
            chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
            dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
            

        
        
        dataset_file_path = os.path.join(output_dir, f"l2_chat/chat_mul_base{base}_raw_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
            
        
        dataset_file_path = os.path.join(output_dir, f"l2/mul_base{base}_raw_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

        
        
        
            
        
        
        
        
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        raw_chat_dataset = []
        raw_dataset = []
        chat_dataset = []
        dataset = []
        for i in range(num_data):
            num_random_op = 1
            random_op_idx = random.sample(range(len(base_function_list)), num_random_op)
            selected_functions = [base_function_list[idx] for idx in random_op_idx]
            random_op_char =  [selected_function['name'] for selected_function in selected_functions]
            ops_dict = dict(zip(random_op_char, selected_functions))
            
            
            example = get_example(ops_dict, random_integers_mid, test=False, type='base', length='var', base=base)
            example, question_ans= example[:-1], example[-1:]
            
            unique_ch = unique_characters(question_ans, example)

            ch2token, token2ch = get_random_token_map(unique_ch)
            new_example = map_string_chars(example, ch2token)
            new_question_ans = map_string_chars(question_ans, ch2token)

            raw_chat_prompt = get_prompt(example, question_ans, chat=True, hint=hint)
            raw_prompt = get_prompt(example, question_ans, chat=False, hint=hint)
            chat_prompt = get_prompt(new_example, new_question_ans, chat=True, hint=hint)
            prompt = get_prompt(new_example, new_question_ans, chat=False, hint=hint)

            raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
            raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
            chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
            dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
            

        
        
        dataset_file_path = os.path.join(output_dir, f"l2_chat/chat_base{base}_raw_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
            
        
        dataset_file_path = os.path.join(output_dir, f"l2/base{base}_raw_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

        dataset_file_path = os.path.join(output_dir, f"l4_chat/chat_base{base}_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
            
        
        dataset_file_path = os.path.join(output_dir, f"l4/base{base}_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    base_data_gen(3)
    base_data_gen(4)
    base_data_gen(5)
    

    def fun_dataset(fun_list, name):
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        raw_chat_dataset = []
        raw_dataset = []
        chat_dataset = []
        dataset = []
        for i in range(num_data):
            num_random_op = 1
            random_op_idx = random.sample(range(len(fun_list)), num_random_op)
            selected_functions = [fun_list[idx] for idx in random_op_idx]
            random_op_char =  [selected_function['name'] for selected_function in selected_functions]
            ops_dict = dict(zip(random_op_char, selected_functions))
            
            
            example = get_example(ops_dict, [random_integers_small, param_intsers_small], test=False, type='linear', length='var')
            example, question_ans= example[:-1], example[-1:]
            
            unique_ch = unique_characters(question_ans, example)

            
            
            

            raw_chat_prompt = get_prompt(example, question_ans, chat=True)
            raw_prompt = get_prompt(example, question_ans, chat=False)
            
            

            raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
            raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
            
            
            

        
        
        dataset_file_path = os.path.join(output_dir, f"l3_chat/chat_{name}_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)
            
        
        dataset_file_path = os.path.join(output_dir, f"l3/{name}_dataset.json")
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_dataset, f, ensure_ascii=False, indent=4)

    fun_dataset(linear_function_list, 'linear')
    fun_dataset(quadratic_function_list, 'quadratic')
    
    
    fun_dataset(sine_function_list, 'sine')
    fun_dataset(cosine_function_list, 'cosine')
    fun_dataset(square_wave_function_list, 'square_wave')
    fun_dataset(triangle_wave_function_list, 'triangle_wave')
    fun_dataset(sawtooth_wave_function_list, 'sawtooth_wave')
    



    all_lists = generate_lists(100, 8, 0, max_val_list)
    print('all_lists: ', all_lists[:10])
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(list_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [list_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, all_lists, test=False, type='list')
        example, question_ans= example[:-1], example[-1:]
        

        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_list_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_list_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_list_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/var_len_list_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l4/var_len_list_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_list_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_list_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_list_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

        SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(list_cnt_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [list_cnt_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, all_lists, test=False, type='list')
        example, question_ans= example[:-1], example[-1:] 
        

        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_list_cnt_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_list_cnt_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_list_cnt_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/var_len_list_cnt_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l4/var_len_list_cnt_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_list_cnt_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_list_cnt_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_list_cnt_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)


    all_datas = generate_date_list(1000, 2010, 2025)
    print("all_datas:", all_datas[:10])
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(data_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [data_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, all_datas, test=False, type='list')
        example, question_ans= example[:-1], example[-1:]
        

        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_data_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/var_len_chat_data_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_data_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/var_len_data_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l4/var_len_data_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_data_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/var_len_data_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/var_len_chat_data_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    raw_chat_dataset = []
    raw_dataset = []
    chat_dataset = []
    dataset = []
    num_chat_dataset = []
    num_dataset = []
    op_chat_dataset = []
    op_dataset = []
    for i in range(num_data):
        num_random_op = 1
        random_op_idx = random.sample(range(len(substr_function_list)), num_random_op)
        random_op_char_idx = random.sample(range(len(special_tokens)), num_random_op)
        random_op_char = [special_tokens[idx] for idx in random_op_char_idx]
        selected_functions = [substr_function_list[idx] for idx in random_op_idx]
        random_op_char =  [selected_function['name'] for selected_function in selected_functions]
        ops_dict = dict(zip(random_op_char, selected_functions))

        example = get_example(ops_dict, [random_strs, random_strs_small], test=False, type='substr')
        example, question_ans= example[:-1], example[-1:]
        

        unique_ch = unique_characters(question_ans, example)
        op_ch = set(random_op_char[0])
        num_ch = unique_ch - op_ch


        ch2token, token2ch = get_random_token_map(unique_ch)
        op2token, token2op = get_random_token_map(op_ch)
        num2token, token2num = get_random_token_map(num_ch)

        new_example = map_string_chars(example, ch2token)
        new_question_ans = map_string_chars(question_ans, ch2token)

        num_example = map_string_chars(example, num2token)
        num_question_ans = map_string_chars(question_ans, num2token)

        op_example = map_string_chars(example, op2token)
        op_question_ans = map_string_chars(question_ans, op2token)

        raw_chat_prompt = get_prompt(example, question_ans, chat=True)
        raw_prompt = get_prompt(example, question_ans, chat=False)
        chat_prompt = get_prompt(new_example, new_question_ans, chat=True)
        prompt = get_prompt(new_example, new_question_ans, chat=False)
        num_chat_prompt = get_prompt(num_example, num_question_ans, chat=True)
        num_prompt = get_prompt(num_example, num_question_ans, chat=False)
        op_chat_prompt = get_prompt(op_example, op_question_ans, chat=True)
        op_prompt = get_prompt(op_example, op_question_ans, chat=False)


        raw_chat_dataset.append({'prompt': raw_chat_prompt, 'answer': question_ans[0][1]})
        raw_dataset.append({'prompt': raw_prompt, 'answer': question_ans[0][1]})
        chat_dataset.append({'prompt': chat_prompt, 'answer': new_question_ans[0][1]})
        dataset.append({'prompt': prompt, 'answer': new_question_ans[0][1]})
        num_chat_dataset.append({'prompt': num_chat_prompt, 'answer': num_question_ans[0][1]})
        num_dataset.append({'prompt': num_prompt, 'answer': num_question_ans[0][1]})
        op_chat_dataset.append({'prompt': op_chat_prompt, 'answer': op_question_ans[0][1]})
        op_dataset.append({'prompt': op_prompt, 'answer': op_question_ans[0][1]})
        

    

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_substr_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l1_chat/fixed_len_chat_substr_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_substr_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    dataset_file_path = os.path.join(output_dir, "l1/fixed_len_substr_raw_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, ensure_ascii=False, indent=4)
    
    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_substr_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_substr_num_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(num_chat_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4/fixed_len_substr_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_dataset, f, ensure_ascii=False, indent=4)

    dataset_file_path = os.path.join(output_dir, "l4_chat/fixed_len_chat_substr_op_dataset.json")
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(op_chat_dataset, f, ensure_ascii=False, indent=4)




    copy_files(os.path.join(output_dir, "l4_chat"), os.path.join(output_dir, "l5_chat"))
    copy_files(os.path.join(output_dir, "l4"), os.path.join(output_dir, "l5"))