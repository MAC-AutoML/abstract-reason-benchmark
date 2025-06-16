from transformers import pipeline
import signal
import torch
import openai

openai.api_base = ''
openai.api_key = ""
openai_model_name = 'gpt-4o-mini'

def timeout_handler(signum, frame):
    raise Exception("Checkout Timeout. Need manual check.")

class LocalAnswerCheckerWithPipeline:
    def __init__(self, model_name, device="cuda"):
        
        self.pipeline = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device=0 if device == "cuda" else -1)
        self.device = device

    def generate_response(self, prompt, max_new_tokens=32):
        
        i = 0
        while i < 30:
            response = self.pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=self.pipeline.tokenizer.eos_token_id)
            if response[0]["generated_text"] and response[0]["generated_text"] > 0:
                return response[0]["generated_text"]
            else:
                i += 1
        

    def check_answer(self, problem, reply_with_answer, ground_truth_answer):
        
        
        message_to_check = (
            "You are a helpful AI assistant. You will use your coding and language skills to verify the answer.\n"
            "You are given:\n"
            "1. A problem.\n"
            "2. A reply with the answer to the problem.\n"
            "3. A ground truth answer.\n"
            "Please do the following:\n"
            "1. Extract the answer in the reply: \"The answer is <answer extracted>\". When extracting, please adhere to the following rules:\n"
            "    - If the answer is a decimal, extract only up to two decimal places (e.g., 4.1234 becomes 4.12).\n"
            "    - If the answer contains commas as thousands separators, remove them.\n"
            "    - If the answer includes units, remove the units.\n"
            "2. Check whether the answer in the reply matches the ground truth answer.\n"
            "3. After everything is done, please choose a reply from the following options (Only one):\n"
            "   - \"The answer is correct.\"\n"
            "   - \"The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>.\"\n"
            "   - \"The reply doesn't contain an answer.\"\n\n"
            f"Problem: {problem}\n\nReply: {reply_with_answer}\n\nGround truth answer: {ground_truth_answer}\n\n"
            "Judje result: "
        )

        
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(300)
            response = self.generate_response(message_to_check)
            signal.alarm(0)
        except Exception as e:
            print(f"Got error: {e}, take it as wrong", flush=True)
            response = "The answer needs manual check."

        
        if "judje result:" in response.lower():
            check_result = response.lower().split("judje result: ")[-1].strip()
        else:
            check_result = response.strip()  
        return {
            "check_result": check_result,
            "is_correct": "the answer is correct" in check_result
            or "the answer is approximated but should be correct" in check_result,
        }

class OpenAIAnswerChecker:
    def __init__(self, model_name=openai_model_name):
        self.model_name = model_name
        self.error_analysis_model = "gpt-4o-mini"  

    def timeout_handler(self, signum, frame):
        raise TimeoutError("OpenAI API request timed out")
    

    def _analyze_error(self, problem, reply_with_answer, ground_truth_answer, response_text):
        
        error_analysis_prompt = [
            {"role": "system", "content":"You are an expert in analyzing why a language model might fail to answer a question correctly. You will receive the problem, the model's reply, the ground truth answer, and the model's output. Based on this, determine the likely cause of the error. Choose one of the following reasons:\n"
                                        "1. The model did not find a pattern in the problem and provided a random output.\n"
                                        "2. The model made a calculation error and did not derive the correct answer.\n"
                                        "3. The model's output process broke down.\n"
                                        "4. Other reasons."
                                        },
            {"role": "user", "content": f"Problem: {problem}\nModel's Reply: {reply_with_answer}\nGround Truth Answer: {ground_truth_answer}\nCheck Result: {response_text}\nError Analysis Reason: "}
        ]

        signal.signal(signal.SIGALRM, self.timeout_handler)
        try:
            signal.alarm(300)
            error_response = openai.ChatCompletion.create(
                model=self.error_analysis_model,
                messages=error_analysis_prompt
            )
            signal.alarm(0)
            error_reason = error_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error Analysis OpenAI Error: {e}, default to 'Other reasons'.", flush=True)
            error_reason = "Other reasons"
        return error_reason

    def check_answer(self, problem, reply_with_answer, ground_truth_answer):
        
        prompt = [
            {"role": "system", "content": 
                "You are a helpful AI assistant. You will use your coding and language skills to verify the answer.\n"
                "You are given:\n"
                "1. A problem.\n"
                "2. A reply with the answer to the problem.\n"
                "3. A ground truth answer.\n"
                "Please do the following:\n"
                "1. Extract the answer in the reply: \"The answer is <answer extracted>\". When extracting, please adhere to the following rules:\n"
                "    - If the answer is a decimal, extract only up to two decimal places (e.g., 4.1234 becomes 4.12).\n"
                "    - If the answer contains commas as thousands separators, remove them.\n"
                "    - If the answer includes units, remove the units.\n"
                "2. Check whether the answer in the reply matches the ground truth answer. When comparing decimal numbers, truncate both the extracted answer and the ground truth answer to two decimal places before comparison.\n"
                "3. After everything is done, please choose a reply from the following options (Only one):\n"
                "   - \"The answer is correct.\"\n"
                "   - \"The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>.\"\n"  
                "   - \"The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>.\"\n"
                "   - \"The reply doesn't contain an answer.\"\n\n"},
            {"role": "user", "content": f"Problem: {problem}\n\nReply: {reply_with_answer}\n\nGround truth answer: {ground_truth_answer}\n\nJudje result: "}
        ]

        signal.signal(signal.SIGALRM, self.timeout_handler)
        try:
            signal.alarm(300)
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=prompt
            )
            signal.alarm(0)
            response_text = response.choices[0].message.content.lower()
        except TimeoutError as e:
            print(f"OpenAI Error: {e}, take it as wrong", flush=True)
            response_text = "The answer needs manual check."
            return {
                "check_result": response_text,
                "is_correct": False,
                "error_reason": "OpenAI API request timed out"
            }
        except Exception as e:
            print(f"OpenAI Error: {e}, take it as wrong", flush=True)
            response_text = "The answer needs manual check."
            return {
                "check_result": response_text,
                "is_correct": False,
                "error_reason": "Other OpenAI Error."
            }


        is_correct = "the answer is correct" in response_text or "the answer is approximated but should be correct" in response_text
        error_reason = None

        
        

        return {
            "check_result": response_text,
            "is_correct": is_correct,
            "error_reason": error_reason
        }

def main():
    
    use_openai = True  

    if use_openai:
        checker = OpenAIAnswerChecker()
        model_name_display = openai_model_name
    else:
        
        local_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        checker = LocalAnswerCheckerWithPipeline(model_name=local_model_name)
        model_name_display = local_model_name

    print(f"Using model: {model_name_display} for answer checking.")

    
    test_cases = [
        {
            "problem": "What is the capital of France?",
            "reply_with_answer": "The answer is Paris.",
            "ground_truth_answer": "Paris",
        },
        {
            "problem": "What is answer?",
            "reply_with_answer": "The answer is 124124.",
            "ground_truth_answer": "124134",
        },
        {
            "problem": "What is answer?",
            "reply_with_answer": "The answer is abcadad.",
            "ground_truth_answer": "abcadad",
        },
        {
            "problem": "What is the largest planet in the solar system?",
            "reply_with_answer": "The answer is Sun.",
            "ground_truth_answer": "Jupiter",
        },
    ]

    
    for i, test in enumerate(test_cases):
        print(f"Test Case {i + 1}:")
        result = checker.check_answer(
            problem=test["problem"],
            reply_with_answer=test["reply_with_answer"],
            ground_truth_answer=test["ground_truth_answer"],
        )
        print(result)
        print("-" * 50)

if __name__ == "__main__":
    main()