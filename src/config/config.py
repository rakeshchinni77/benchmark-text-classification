"""Centralized configuration for the benchmark-text-classification project.

This module contains only configuration values and safe runtime detection.
It intentionally avoids business logic so the rest of the application can
import a single source of truth for paths, dataset names, model identifiers,
and benchmark parameters.
"""

from __future__ import annotations

from pathlib import Path

try:
	import torch
except ImportError:  # pragma: no cover - optional dependency during Phase 2
	torch = None


# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root."""

SRC_DIR: Path = PROJECT_ROOT / "src"
"""Absolute path to the source directory."""

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
"""Directory where experiment artifacts are written."""

RESULTS_DIR: Path = PROJECT_ROOT / "results"
"""Directory that stores benchmark outputs."""

MODEL_OUTPUT_DIR: Path = OUTPUT_DIR / "models"
"""Directory for saved model checkpoints and fine-tuned artifacts."""

LOG_DIR: Path = OUTPUT_DIR / "logs"
"""Directory for application and experiment logs."""

PLOTS_DIR: Path = OUTPUT_DIR / "plots"
"""Directory for generated charts and comparison plots."""

RESULTS_JSON_PATH: Path = RESULTS_DIR / "results.json"
"""Canonical JSON output path for benchmark results."""


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------

DATASET_NAME: str = "ag_news"
"""Name of the Hugging Face dataset used for benchmarking."""

LABEL_NAMES: list[str] = [
	"World",
	"Sports",
	"Business",
	"Science and Technology",
]
"""Human-readable label names aligned with the dataset label ids."""


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"
"""Model used for zero-shot classification."""

FEW_SHOT_MODEL: str = "google/flan-t5-base"
"""Model used for few-shot prompting experiments."""

FINE_TUNE_MODEL: str = "distilbert-base-uncased"
"""Base encoder used for supervised fine-tuning."""


# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

K_VALUES: list[int] = [2, 4, 8, 16]
"""Number of in-context examples to test for few-shot experiments."""

SEEDS: list[int] = [42, 123, 999]
"""Random seeds used for sensitivity and reproducibility experiments."""

TRAIN_SUBSETS: list[int] = [100, 500, 2000, 10000, 120000]
"""Training subset sizes used in the fine-tuning data-efficiency study."""


# ---------------------------------------------------------------------------
# Runtime Configuration
# ---------------------------------------------------------------------------

def _detect_device() -> str:
	"""Return a safe execution device name for the current environment.

	The function prefers CUDA when PyTorch is installed and reports that a GPU
	is available. If PyTorch is unavailable or CUDA cannot be used safely, the
	runtime falls back to CPU.
	"""

	if torch is None:
		return "cpu"

	try:
		return "cuda" if torch.cuda.is_available() else "cpu"
	except Exception:
		return "cpu"


DEVICE: str = _detect_device()
"""Execution device detected at import time using safe fallback logic."""


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

NUM_EPOCHS: int = 3
"""Number of fine-tuning epochs."""

TRAIN_BATCH_SIZE: int = 32
"""Training batch size used by the Trainer configuration."""

EVAL_BATCH_SIZE: int = 64
"""Evaluation batch size used by the Trainer configuration."""

LEARNING_RATE: float = 2e-5
"""Learning rate for supervised fine-tuning."""

MAX_LENGTH: int = 512
"""Maximum tokenized sequence length for model inputs."""


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------

FEW_SHOT_EVAL_SIZE: int = 500
"""Number of test examples used for few-shot evaluation."""

FULL_TEST_SIZE: int = 7600
"""Full AG News test split size used for standard evaluation."""
