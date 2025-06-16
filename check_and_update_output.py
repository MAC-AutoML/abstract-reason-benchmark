import json
import os
import argparse
from tqdm import tqdm
from check_pipeline import LocalAnswerCheckerWithPipeline, OpenAIAnswerChecker
import re

name_map = {
    'l0':'Basic Computation Tasks',
    'l1':'Extended Calculation Tasks',
    'l2':'Number Base Reasoning Tasks',
    'l3':'Symbolic Mathematical Abstraction Tasks',
    'l4':'Symbolic Reasoning Tasks',
    'l6':'Mathematical Application Tasks',
}

def extract_last_num(text: str) -> float:
    
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    
    res = re.findall(r"-?\d+(?:\.\d+)?", text)

    if not res:  
        return None

    return float(res[-1])  

def extract_answer(text):
    
    pattern = r"The answer is (.*?)(?:\.\s|\.\n|$)"
    match = re.search(pattern, text)
    if match:
        
        answer = match.group(1)
        
        answer = re.sub(r'[^\w\s\.,\{\}\[\]-]', '', answer)
        answer = answer.rstrip('.')
        return answer
    return ''

def check_answer_with_checker(prompt, output_text, ground_truth_answer, checker):
    
    result = checker.check_answer(
        problem=prompt,
        reply_with_answer=output_text,
        ground_truth_answer=ground_truth_answer,
    )
    return result["is_correct"], result

def add_averages_to_level_result(level_result):
    total_sum_all = 0
    count_all = 0
    total_sum_part = 0
    count_part = 0
    total_sum_op = 0
    count_op = 0
    total_sum_num = 0
    count_num = 0
    keys_to_exclude = ['_square_', '_data_', '_op_', '_num_']

    for key, value in level_result.items():
        
        total_sum_all += value
        count_all += 1

        
        exclude = False
        for pattern in keys_to_exclude:
            if pattern in key:
                exclude = True
                break
        if not exclude:
            total_sum_part += value
            count_part += 1

        
        if '_op_' in key:
            total_sum_op += value
            count_op += 1

        
        if '_num_' in key:
            total_sum_num += value
            count_num += 1

    
    avg_all = total_sum_all / count_all if count_all > 0 else 0
    avg_part = total_sum_part / count_part if count_part > 0 else 0
    avg_op = total_sum_op / count_op if count_op > 0 else 0
    avg_num = total_sum_num / count_num if count_num > 0 else 0

    
    level_result['avg_all'] = avg_all
    level_result['avg_part'] = avg_part
    level_result['avg_op'] = avg_op
    level_result['avg_num'] = avg_num
    return level_result

def calculate_value_percentage(error_reason_result):


    total_sum = sum(error_reason_result.values())
    if total_sum == 0:
        return {key: 0.00 for key in error_reason_result}  

    percentage_result = {}
    for key, value in error_reason_result.items():
        percentage = (value / total_sum) * 100
        percentage_result[key] = round(percentage, 2) 
    return percentage_result

def process_output_files(base_dir, checker):
    
    total_input_tokens = 0  
    total_output_tokens = 0  

    levels = ['l0', 'l1', 'l2', 'l3', 'l4', 'l6']  
    
    all_results_llm = {}
    error_reason_result = {  
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0,
    }
    final_result = {}
    for level in levels:
        level_result = {}
        llm_level_result = {}
        error_reason_result_level = { 
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
        }
        level_dir = os.path.join(base_dir, level + '_chat')
        
        if not os.path.exists(level_dir):
            print(f"Level directory {level_dir} does not exist. Skipping.")
            continue
        datasets_path = [f for f in os.listdir(level_dir) if 'dataset' in f]  
        for dataset_file in datasets_path:
            dataset_file_path = os.path.join(level_dir, dataset_file)
            output_path = os.path.join(dataset_file_path, 'output.json')
            result_path = os.path.join(dataset_file_path, 'result.json')
            
            print('-'*30)
            print(output_path)
            with open(output_path, 'r') as f:
                dataset = json.load(f)

            total_correct_llm = 0  
            total_correct_em = 0   
            total_correct_em_strict = 0  
            total_count = 0        
            total_input_tokens_dataset = 0
            total_output_tokens_dataset = 0

            
            for record in tqdm(dataset, desc=f"Processing {output_path}"):
                prompt = record["prompt"][-1]['content']  
                prompt = prompt.split('\n')[-1]
                record["output"] = extract_answer(record["raw_output"])
                output_text = record["output"]
                ground_truth_answer = record["answer"]

                
                is_correct_llm = None
                is_correct_em = False
                is_correct_em_strict = False
                total_count += 1

                if checker and ('llm_judge' not in record or 'llm_judge_result' not in record or len(record['llm_judge_result']) < 3 ):
                
                    
                    is_correct_llm, result = check_answer_with_checker(prompt, record["raw_output"], ground_truth_answer, checker)
                    record['llm_judge'] = is_correct_llm
                    record['llm_judge_result'] = result
                if record['llm_judge_result']["error_reason"]:
                    record['llm_judge_result']["error_reason"]
                    try:
                        reason_str = str(int(extract_last_num(record['llm_judge_result']["error_reason"])))
                        if reason_str not in ['1','2', '3', '4']:
                            reason_str = '4'
                    except:
                        reason_str = '4'

                    error_reason_result_level[reason_str] = error_reason_result_level[reason_str] + 1
                    error_reason_result[reason_str] = error_reason_result[reason_str] + 1
                    
                if record['llm_judge']:
                    total_correct_llm += 1
                
                if extract_last_num(ground_truth_answer) is not None and ('l0' in level or 'l3' in level):
                    
                    if extract_last_num(output_text) is not None:
                        is_correct_em = abs(extract_last_num(output_text) - extract_last_num(ground_truth_answer)) < 1e-2
                    else:
                        is_correct_em = False
                else:
                    
                    is_correct_em = (ground_truth_answer in output_text)

                
                record['EM_result'] = is_correct_em

                
                if extract_last_num(ground_truth_answer) is not None and ('l0' in level or 'l3' in level):
                    
                    if extract_last_num(output_text) is not None:
                        is_correct_em_strict = abs(extract_last_num(output_text) - extract_last_num(ground_truth_answer)) < 1e-2
                    else:
                        is_correct_em_strict = False
                else:
                    
                    is_correct_em_strict = (ground_truth_answer == output_text)
                
                record['EM_strict_result'] = is_correct_em_strict

                
                
                

                if is_correct_em:
                    total_correct_em += 1
                if is_correct_em_strict:
                    total_correct_em_strict += 1
                if "input_tokens" in record:
                    total_input_tokens_dataset += record["input_tokens"]
                    total_output_tokens_dataset += record["output_tokens"]
            
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=4)

            print(f"Updated results saved to {output_path}")

            
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result_data = json.load(f)

                
                accuracy_llm = total_correct_llm / total_count if total_count > 0 else 0
                accuracy_em = total_correct_em / total_count if total_count > 0 else 0
                accuracy_em_strict = total_correct_em_strict / total_count if total_count > 0 else 0
                result_data["result_judge_by_llm"] = accuracy_llm
                result_data["result_judge_by_em"] = accuracy_em
                result_data["result_judge_by_em_strict"] = accuracy_em_strict
                result_data["total_count"] = total_count
                result_data["token_usage"] = {
                    "total_input_tokens" : total_input_tokens_dataset,
                    "total_output_tokens": total_output_tokens_dataset
                }
                
                
                if isinstance(result_data["token_usage"], dict):
                    total_input_tokens += result_data["token_usage"]["total_input_tokens"]
                    total_output_tokens += result_data["token_usage"]["total_output_tokens"]
                else:
                    total_input_tokens += result_data["token_usage"]

                with open(result_path, 'w') as f:
                    json.dump(result_data, f, indent=4)
                level_result[dataset_file] = accuracy_em_strict
                llm_level_result[dataset_file] = accuracy_llm
                all_results_llm[dataset_file] = accuracy_llm

        
        error_reason_result_percentage_level = calculate_value_percentage(error_reason_result_level)
        
        error_reason_path = os.path.join(level_dir, 'error_reason.json')
        with open(error_reason_path, 'w') as f:
            json.dump(error_reason_result_percentage_level, f, indent=4)
        print(f"Saved error reasons for {level} to {error_reason_path}")

        level_result = add_averages_to_level_result(level_result)
        level_result_path = os.path.join(level_dir, 'result.json')
        level_result_sorted = {key: level_result[key] for key in sorted(level_result.keys())}

        
        level_result_path = os.path.join(level_dir, 'result.json')
        with open(level_result_path, 'w') as f:
            json.dump(level_result_sorted, f, indent=4)

        llm_level_result = add_averages_to_level_result(llm_level_result)
        llm_level_result_path = os.path.join(level_dir, 'llm_result.json')
        llm_level_result_sorted = {key: llm_level_result[key] for key in sorted(llm_level_result.keys())}

        
        llm_level_result_path = os.path.join(level_dir, 'llm_result.json')
        with open(llm_level_result_path, 'w') as f:
            json.dump(llm_level_result_sorted, f, indent=4)

        final_result[name_map[level]] = llm_level_result_sorted["avg_all"]
        if(level == 'l1'):
            final_result["with_mem"] = llm_level_result_sorted["avg_part"]
        if(level == 'l4'):
            final_result["without_all_mem"] = llm_level_result_sorted["avg_part"]
            final_result["without_op_mem"] = llm_level_result_sorted["avg_op"]
            final_result["without_num_mem"] = llm_level_result_sorted["avg_num"]
            final_result["memdep_all"] = final_result["with_mem"] - final_result["without_all_mem"]
            final_result["memdep_op"] = final_result["with_mem"] - final_result["without_op_mem"]
            final_result["memdep_num"] = final_result["with_mem"] - final_result["without_num_mem"]

    
    token_usage = {"total_input_tokens": total_input_tokens,
                   "total_output_tokens": total_output_tokens}
    print(base_dir + '/token_usage.json')
    with open(base_dir + '/token_usage.json', 'w') as f:
        json.dump(token_usage, f, indent=4)

    avg_result_llm = add_averages_to_level_result(all_results_llm)
    with open(base_dir + '/all_result_llm.json', 'w') as f:
        json.dump(avg_result_llm, f, indent=4)

    
    error_reason_result_percentage = calculate_value_percentage(error_reason_result)
    print(error_reason_result_percentage)
    with open(base_dir + '/error_reason.json', 'w') as f:
        json.dump(error_reason_result_percentage, f, indent=4)
    
    final_result['avg_all'] = avg_result_llm["avg_all"]
    final_result_sort = {
        "Basic Computation Tasks":final_result["Basic Computation Tasks"],
        "Extended Calculation Tasks":final_result["Extended Calculation Tasks"],
        "Number Base Reasoning Tasks":final_result["Number Base Reasoning Tasks"],
        "Mathematical Application Tasks":final_result["Mathematical Application Tasks"],
        "Symbolic Mathematical Abstraction Tasks":final_result["Symbolic Mathematical Abstraction Tasks"],
        "Symbolic Reasoning Tasks":final_result["Symbolic Reasoning Tasks"],
        "avg_all":final_result["avg_all"],
        "memdep_op":final_result["memdep_op"],
        "memdep_num":final_result["memdep_num"],
        "memdep_all":final_result["memdep_all"]
    }
    with open(base_dir + '/final_result.json', 'w') as f:
        json.dump(final_result_sort, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process output files and evaluate results.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for processing.")
    parser.add_argument("--llm_judge", action="store_true", help="Enable LLM judgment checking.")
    args = parser.parse_args()

    
    checker = None
    if args.llm_judge:
        
        checker = OpenAIAnswerChecker()

    process_output_files(args.base_dir, checker)