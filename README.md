# Benchmarking Abstract and Reasoning Abilities Through A Theoretical Perspective

This repository hosts the official code, datasets, and results for our paper:

**Benchmarking Abstract and Reasoning Abilities Through A Theoretical Perspective**

*Qingchuan Ma*, *Yuhang Wu*, *Xiawu Zheng*, *Rongrong Ji*

Accepted at the **42nd International Conference on Machine Learning (ICML 2025)**.

## Abstract

In this paper, we aim to establish a simple, effective, and theoretically grounded benchmark for rigorously probing abstract reasoning in Large Language Models (LLMs). To achieve this, we first develop a mathematic framework that defines abstract reasoning as the ability to: (i) extract essential patterns independent of surface representations, and (ii) apply consistent rules to these abstract patterns. Based on this framework, we introduce two novel complementary metrics: Γ measures basic reasoning accuracy, while Δ quantifies a model’s reliance on specific symbols rather than underlying patterns - a key indicator of true abstraction versus mere memorization. To implement this measurement, we design a benchmark: systematic symbol remapping in rule-based tasks, which forces models to demonstrate genuine pattern recognition beyond superficial token matching. Extensive LLM evaluations using this benchmark (commercial API models, 7B-70B, multi-agent) reveal: 1) critical limitations in non-decimal arithmetic and symbolic reasoning; 2) persistent abstraction gaps despite chain-of-thought prompting; and 3) Δ’s effectiveness in robustly measuring memory dependence by quantifying performance degradation under symbol remapping, particularly highlighting operand-specific memorization. These findings underscore that current LLMs, despite domain-specific strengths, still lack robust abstract reasoning, highlighting key areas for future improvement.

## Setup

1. **Clone the repository:**
   
   ```bash
   git clone git@github.com:MAC-AutoML/abstract-reason-benchmark.git
   cd abstract-reason-benchmark
   ```

## Usage

This repository provides scripts for dataset generation, model evaluation, and results analysis.

### Step 1: Dataset Generation

You can generate your own custom dataset with different symbol mappings and rules.

```bash
python main.py
```

This will create a new dataset directory. For convenience, we have already provided a pre-generated dataset in the `dataset_1129` folder.

### Step 2: Running Evaluation

We support evaluation for both locally-hosted models and API-based models.

#### A. Testing Local Models (e.g., Llama 3.1 8B)

Use `batch_test.py` to evaluate models hosted locally (e.g., via Hugging Face `transformers`).

**Standard Evaluation:**

```bash
CUDA_VISIBLE_DEVICES=0 python batch_test.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --chat \
    --batch_size 2 \
    --dataset dataset_1129
```

**Chain-of-Thought (CoT) Evaluation:**

```bash
CUDA_VISIBLE_DEVICES=0 python batch_test.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --chat \
    --batch_size 2 \
    --cot \
    --cot_type zero \
    --dataset dataset_1129
```

* `--model_name`: The model identifier from Hugging Face.
* `--chat`: Use the model's chat template.
* `--cot`: Enable Chain-of-Thought prompting.
* `--dataset`: Path to the evaluation dataset.

#### B. Testing API-based Models (e.g., GPT-4o-mini)

Use `test_openai.py` to evaluate models accessible via an API endpoint.

```bash
python test_openai.py \
    --model_name gpt-4o-mini \
    --key YOUR_API_KEY \
    --base_url YOUR_API_BASE_URL
```

* Replace `YOUR_API_KEY` and `YOUR_API_BASE_URL` with your credentials.

### Step 3: Judging the Outputs (LLM-as-Judge)

After generating model outputs, use our script `check_and_update_output.py` to parse the results and score them using a powerful LLM judge (e.g., GPT-4o-mini).

```bash
# Judge standard evaluation results
python check_and_update_output.py --base_dir gpt-4o-mini/result --llm_judge

# Judge CoT evaluation results
python check_and_update_output.py --base_dir gpt-4o-mini/result_cot --llm_judge
```

* `--base_dir`: The directory containing the model's raw output files.
* `--llm_judge`: Flag to enable the LLM judge for scoring. You will need to configure your API key for the judge model inside the script.

## Our Metrics: Γ and Δ

The evaluation scripts will compute our two proposed metrics based on the scored results:

* **Γ (Abstract Reasoning Score):** Measures the accuracy on the reasoning tasks.
* **Δ (Memory Dependence Score):** Quantifies the performance degradation when symbols are remapped, indicating the model's reliance on memorized symbols versus abstract patterns. A higher Δ score signifies a larger abstraction gap.

## Repository Structure

```
.
├── dataset_1129/               # Pre-generated benchmark dataset
├── gpt-4o-mini/                  # Example evaluation results for GPT-4o-mini
│   ├── result/                   # Raw outputs from standard evaluation
│   └── result_cot/               # Raw outputs from CoT evaluation
├── main.py                       # Script to generate datasets
├── batch_test.py                 # Script to evaluate local models
├── test_openai.py                # Script to evaluate API-based models
├── check_and_update_output.py    # Script for LLM-as-judge scoring
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Citation

If you find our work useful, please consider citing our paper:

```
@article{ma2025benchmarking,
  title={Benchmarking Abstract and Reasoning Abilities Through A Theoretical Perspective},
  author={Ma, Qingchuan and Wu, Yuhang and Zheng, Xiawu and Ji, Rongrong},
  journal={arXiv preprint arXiv:2505.23833},
  year={2025}
}
```

Thank you for your interest in our work!

