
# python main.py

CUDA_VISIBLE_DEVICES=0 python batch_test.py --model_name meta-llama/Llama-3.1-8B-Instruct --chat --batch_size 2 --dataset dataset_1129
CUDA_VISIBLE_DEVICES=0 python batch_test.py --model_name meta-llama/Llama-3.1-8B-Instruct --chat --batch_size 2  --cot --cot_type zero --dataset dataset_1129




python test_openai.py --model_name gpt-4o-mini --key API_KEY --base_url Base_url


# python check_and_update_output.py --base_dir gpt-4o-mini/result --llm_judge
# python check_and_update_output.py --base_dir gpt-4o-mini/result_cot --llm_judge

