# PROJECT_CONTEXT.md

## Project Name

Benchmark Text Classification with Hugging Face:
Zero-Shot vs Few-Shot vs Fine-Tuning

---

# Project Objective

Build a production-grade NLP benchmarking system that compares three text classification approaches on the AG News dataset:

1. Zero-Shot Classification
2. Few-Shot Classification
3. Fine-Tuned Classification

The project must satisfy all evaluation requirements and generate:

results/results.json

with the exact schema specified in the task requirements.

The project must be runnable via:

docker-compose up --build

and eventually execute the complete benchmarking pipeline automatically.

---

# Dataset

Dataset Name:

ag_news

Source:

Hugging Face Datasets Library

Classes:

0 → World

1 → Sports

2 → Business

3 → Science and Technology

Dataset Size:

Training Set: 120,000

Test Set: 7,600

---

# Repository Structure

benchmark-text-classification/

docker-compose.yml

Dockerfile

.env.example

requirements.txt

README.md

src/

config/

data/

models/

evaluation/

experiments/

utils/

results/

outputs/

models/

logs/

plots/

results/

results.json

---

# Architecture

Workflow:

AG News Dataset

↓

Data Loader

↓

Benchmark Methods

├── Zero-Shot

├── Few-Shot

└── Fine-Tuned

↓

Evaluation Engine

↓

Results Aggregation

↓

results/results.json

---

# Core Models

## Zero-Shot

Model:

facebook/bart-large-mnli

Purpose:

Classify AG News without training.

---

## Few-Shot

Model:

google/flan-t5-base

Purpose:

Prompt-based classification using examples.

Supported k values:

2

4

8

16

Important:

Few-shot prompts must preserve:

Instruction

Examples

Query Text

Final "Category:" label slot

Do NOT redesign prompt format without explicit request.

---

## Fine-Tuning

Model:

distilbert-base-uncased

Framework:

Hugging Face Trainer API

Training:

3 epochs

fp16=True when GPU available

---

# Evaluation Metrics

Accuracy

Macro F1

Median Latency

P99 Latency

Peak GPU Memory

Training Time

Data Efficiency Accuracy

Few-Shot Sensitivity Statistics

---

# Required Experiments

## Zero-Shot Benchmark

Evaluate:

facebook/bart-large-mnli

Output:

accuracy

macro_f1

latency

gpu memory

---

## Few-Shot Benchmark

Evaluate:

k=2

k=4

k=8

k=16

Output:

accuracy

macro_f1

latency

gpu memory

---

## Few-Shot Sensitivity

Run:

k=8

seed=42

seed=123

seed=999

Output:

accuracy_mean

accuracy_std

---

## Fine-Tuning

Train:

distilbert-base-uncased

Evaluate:

accuracy

macro_f1

training time

gpu memory

---

## Data Efficiency

Train on:

100

500

2000

10000

120000

examples

Output:

accuracy curve

Expected trend:

accuracy generally increases as training data increases.

---

# Current Development Strategy

Local Laptop:

Phase 0 → Phase 7

Google Colab:

Phase 8 → Phase 10

Laptop Again:

Phase 11 → Phase 12

Reason:

Laptop has:

Intel i5-12500H

16GB RAM

Intel Iris Xe

No dedicated NVIDIA GPU

Fine-tuning experiments will be executed in Google Colab using T4 GPU.

---

## phases roadmap is

# Local Development Phases

Phase 0 Repository Foundation Output: Professional project structure Phase 1 Docker & Environment Output: docker-compose up --build works Phase 2 Configuration Layer Files: src/config/config.py Output: Single source of truth Phase
3 Data Layer Files: src/data/ Output: AG News loader Label mapping Preprocessing
Phase 4 Evaluation Framework Files: src/evaluation/ Output: Accuracy F1 Latency Memory Universal evaluator
Phase 5 Zero-Shot Benchmark Model: facebook/bart-large-mnli Output: zero_shot_results Phase
6 Few-Shot Benchmark Model: google/flan-t5-base Output: few_shot_results
Phase 7 Few-Shot Sensitivity Output: mean accuracy std accuracy At this point: GitHub Docker Evaluation Engine Zero-Shot Few-Shot Sensitivity all complete. Commit: git commit -m "Phase 7: Complete few-shot benchmarking pipeline" git push 
## Colab Phases
Phase 8 Fine-Tuning DistilBERT Output: finetuned_full model Save: outputs/models/finetuned_full/ Download to laptop.
Phase 9 Inference Benchmarking Run: bs=1 bs=128 Collect: latency metrics
Phase 10 Data Efficiency Train: 100 500 2000 10000 120000 Collect: accuracy curve Export results JSON. 

## Back To Laptop
Phase 11 Results Aggregation Generate: results/results.json using: Zero-Shot results Few-Shot results Sensitivity results Fine-Tuning results Data Efficiency results
Phase 12 README + Plots + Final Packaging Generate: accuracy_comparison.png latency_comparison.png data_efficiency.png Complete README.

---

# Completed Phases

## Phase 0

Repository Foundation

Status:

Completed

---

## Phase 1

Docker & Environment

Status:

Completed

docker-compose up --build works.

---

## Phase 2

Configuration Layer

Status:

Completed

config.py is the single source of truth.

---

## Phase 3

Data Layer

Status:

Completed

AG News loader

Label mapping

Preprocessing

implemented.

---

## Phase 4

Evaluation Framework

Status:

Completed

Accuracy

Macro F1

Latency

GPU Memory

Universal Evaluator

implemented.

---

## Phase 5

Zero-Shot Benchmark

Status:

Completed

facebook/bart-large-mnli implemented.

---

## Phase 6

Few-Shot Benchmark

Status:

Completed

Model:

google/flan-t5-base

Important:

Preserve existing architecture.

Do not rewrite working modules.

Make minimal changes only.

Any fix must preserve benchmark behavior.


Phase 7 - Few-Shot Sensitivity

status - completed


Phase 8 — Fine-Tuning DistilBERT
status - completed
Goal:
Fine-tune DistilBERT on AG News dataset.

Model:
distilbert-base-uncased

Dataset:
AG News

Training:
3 epochs
fp16=True
Trainer API

Outputs:
outputs/models/finetuned_full/

Required Metrics:
training_time_minutes
peak_training_gpu_memory_mb
accuracy
macro_f1

current phase
Phase 9-Inference Benchmarking


Future Dependencies:
Phase 10 uses same pipeline for data efficiency study.

---

# Coding Rules For Copilot

Before modifying code:

1. Read PROJECT_CONTEXT.md.

2. Preserve existing architecture.

3. Do not redesign modules unless explicitly requested.

4. Prefer small fixes over rewrites.

5. Never change public interfaces unless instructed.

6. Keep code production-ready.

7. Keep compatibility with future phases.

8. Ensure outputs remain compatible with results.json schema.

9. Do not remove evaluation metrics.

10. Explain all changes before making them.

---

# End of Context
