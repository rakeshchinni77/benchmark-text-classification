"""Generate benchmark visualizations from aggregated results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.config.config import PLOTS_DIR, RESULTS_JSON_PATH

logger = logging.getLogger(__name__)


def _load_results() -> dict[str, Any]:
    try:
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.warning("Missing results file: %s", RESULTS_JSON_PATH)
        return {}
    except Exception:
        logger.exception("Failed to load results from %s", RESULTS_JSON_PATH)
        return {}


def _best_few_shot_setting(results: dict[str, Any]) -> tuple[str | None, float, float]:
    few_shot = results.get("few_shot", {}) or {}
    best_k = None
    best_accuracy = -1.0
    best_latency = float("inf")

    for key, data in few_shot.items():
        if not key.startswith("k_"):
            continue
        accuracy = data.get("accuracy")
        latency = data.get("median_latency_ms")
        if accuracy is None or latency is None:
            continue
        if accuracy > best_accuracy or (accuracy == best_accuracy and latency < best_latency):
            best_accuracy = accuracy
            best_latency = latency
            best_k = key

    if best_k is None:
        return None, 0.0, 0.0

    return best_k, best_accuracy, best_latency


def _save_figure(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def _plot_accuracy_comparison(results: dict[str, Any]) -> None:
    zero_accuracy = results.get("zero_shot", {}).get("accuracy", 0.0)
    fine_accuracy = results.get("fine_tuning", {}).get("evaluation", {}).get("accuracy", 0.0)
    best_k, best_few_accuracy, _ = _best_few_shot_setting(results)
    few_label = f"Few-Shot ({best_k})" if best_k else "Few-Shot"

    labels = ["Zero-Shot", few_label, "Fine-Tuning"]
    values = [zero_accuracy, best_few_accuracy, fine_accuracy]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["#377eb8", "#4daf4a", "#e41a1c"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center", va="bottom")

    _save_figure(fig, "accuracy_comparison.png")


def _plot_latency_comparison(results: dict[str, Any]) -> None:
    zero_latency = results.get("zero_shot", {}).get("median_latency_ms", 0.0)
    best_k, _, few_latency = _best_few_shot_setting(results)
    fine_latency = results.get("inference", {}).get("batch_size_1", {}).get("per_sample_latency_ms", 0.0)
    few_label = f"Few-Shot ({best_k})" if best_k else "Few-Shot"

    labels = ["Zero-Shot", few_label, "Fine-Tuning"]
    values = [zero_latency, few_latency, fine_latency]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["#377eb8", "#4daf4a", "#e41a1c"])
    ax.set_ylabel("Median Latency (ms)")
    ax.set_title("Latency Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for index, value in enumerate(values):
        ax.text(index, value + max(1.0, value * 0.02), f"{value:.1f}", ha="center", va="bottom")

    _save_figure(fig, "latency_comparison.png")


def _plot_data_efficiency(results: dict[str, Any]) -> None:
    data_efficiency = results.get("data_efficiency", {}) or {}
    sample_sizes = sorted((int(key) for key in data_efficiency.keys()), key=int)
    accuracies = [data_efficiency[str(size)].get("accuracy", 0.0) for size in sample_sizes]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sample_sizes, accuracies, marker="o", color="#377eb8", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Data Efficiency: Training Samples vs Accuracy")
    ax.set_xticks(sample_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)

    for x, y in zip(sample_sizes, accuracies):
        ax.text(x, y + 0.02, f"{y:.3f}", ha="center", va="bottom")

    _save_figure(fig, "data_efficiency.png")


def generate_all_plots() -> None:
    results = _load_results()
    _plot_accuracy_comparison(results)
    _plot_latency_comparison(results)
    _plot_data_efficiency(results)


__all__ = ["generate_all_plots"]
