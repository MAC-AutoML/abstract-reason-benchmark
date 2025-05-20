# Benchmarking Abstract and Reasoning Abilities Through A Theoretical Perspective

This repository will soon host the official code and datasets for our paper:

**Benchmarking Abstract and Reasoning Abilities Through A Theoretical Perspective**

*Qingchuan Ma*, *Yuhang Wu*, *Xiawu Zheng*, *Rongrong Ji*

Accepted at the **42nd International Conference on Machine Learning (ICML 2025)**.

## Abstract

In this paper, we aim to establish a simple, effective, and theoretically grounded benchmark for rigorously probing abstract reasoning in Large Language Models (LLMs). To achieve this, we first develop a mathematic framework that defines abstract reasoning as the ability to: (i) extract essential patterns independent of surface representations, and (ii) apply consistent rules to these abstract patterns. Based on this framework, we introduce two novel complementary metrics: Γ measures basic reasoning accuracy, while Δ quantifies a model’s reliance on specific symbols rather than underlying patterns - a key indicator of true abstraction versus mere memorization. To implement this measurement, we design a benchmark: systematic symbol remapping in rule-based tasks, which forces models to demonstrate genuine pattern recognition beyond superficial token matching. Extensive LLM evaluations using this benchmark (commercial API models, 7B-70B, multi-agent) reveal: 1) critical limitations in non-decimal arithmetic and symbolic reasoning; 2) persistent abstraction gaps despite chain-of-thought prompting; and 3) Δ’s effectiveness in robustly measuring memory dependence by quantifying performance degradation under symbol remapping, particularly highlighting operand-specific memorization. These findings underscore that current LLMs, despite domain-specific strengths, still lack robust abstract reasoning, highlighting key areas for future improvement.

## Coming Soon!

We are currently preparing the code, datasets, and detailed instructions for public release. Please check back soon for updates.

We plan to release:
*   The complete benchmark dataset.
*   Scripts for generating tasks with systematic symbol remapping.
*   Evaluation scripts to compute Γ (Abstract Reasoning Score) and Δ (Memory Dependence Score).
*   Detailed documentation and examples.

Thank you for your interest in our work!
