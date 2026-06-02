# benchmark-text-classification

Production-grade NLP benchmarking scaffold for comparing zero-shot, few-shot, and fine-tuned text classification approaches using the Hugging Face ecosystem.

## Project Overview

This repository is structured to benchmark three common paradigms for text classification:

- Zero-shot classification using a pre-trained NLI model
- Few-shot classification using in-context examples
- Fine-tuning a sequence classification model on a labeled dataset

The project will ultimately produce structured benchmark results, plots, and a decision guide for selecting the best technique under different constraints.

## Problem Statement

Text classification projects often need to balance accuracy, latency, cost, data availability, and operational complexity. This repository is intended to provide a reproducible framework for comparing those trade-offs in a controlled way.

## Objectives

- Establish a reproducible benchmark pipeline for AG News classification
- Compare zero-shot, few-shot, and fine-tuned approaches under a shared evaluation protocol
- Capture latency, accuracy, F1, and memory metrics
- Produce a structured results artifact for automated validation
- Provide a decision guide for future model selection

## Architecture Overview

The repository is organized into clear layers:

- `src/data` for dataset loading and preprocessing
- `src/models` for classification strategy wrappers
- `src/evaluation` for metrics, latency, and memory tracking
- `src/experiments` for benchmark execution flows
- `src/results` for aggregation and JSON export
- `src/utils` for shared helpers

## Folder Structure

- `src/` application source and package modules
- `outputs/` model artifacts, logs, and plots produced during experiments
- `results/` structured benchmark outputs
- `.venv/` local-only virtual environment, ignored by Git

## Planned Experiments

- Zero-shot baseline on the AG News test split
- Few-shot evaluation for `k = 2, 4, 8, 16`
- Few-shot sensitivity study for repeated `k = 8` runs
- Fine-tuning on the full dataset and on smaller data subsets
- Inference benchmarking at batch size 1 and batch size 128

## Expected Results

The final pipeline will generate `results/results.json` containing benchmark metrics for all three approaches, plus summary statistics for sensitivity and data-efficiency experiments.

## Future Phases

- Phase 1: Docker and environment foundation
- Phase 2: Core configuration and utility modules
- Phase 3: Data loading and preprocessing
- Phase 4: Zero-shot implementation and evaluation
- Phase 5: Few-shot prompting and sensitivity analysis
- Phase 6: Fine-tuning pipeline and data efficiency study
- Phase 7: Result aggregation, charts, and final decision guide

## Local Development Setup

Create a virtual environment locally:

```bash
python -m venv .venv
```

Activate it, install dependencies, and prepare your local environment before running the containerized workflow.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust any local Hugging Face cache or GPU settings as needed.

## Docker Setup

Build and run the Phase 1 container bootstrap with:

```bash
docker-compose up --build
```

This startup path currently prints the initialization message and ensures the expected runtime folders exist:

- `outputs/logs`
- `outputs/models`
- `outputs/plots`
- `results`

The repository also includes a `.dockerignore` file to keep the build context small and predictable. It excludes the local virtual environment, Git metadata, editor settings, Python caches, Hugging Face caches, and generated model/results artifacts so Docker does not spend time sending unnecessary files to the daemon.

That keeps `docker-compose build` faster and prevents local-only files from being copied into the image.

Phase 1 intentionally keeps `requirements.txt` lightweight so the base container can build quickly without downloading the full machine learning stack. The Hugging Face, PyTorch, and model-training dependencies will be added in later phases when the benchmark pipeline itself is implemented.

The container is prepared for later ML implementation phases, including Hugging Face caching and future GPU-enabled training.

## Docker Setup Placeholder

Docker support is scaffolded at the repository root with `Dockerfile` and `docker-compose.yml`. The full experimental pipeline will be wired up in a later phase.
