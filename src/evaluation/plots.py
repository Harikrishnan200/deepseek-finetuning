"""Plot generation. Matplotlib only, no seaborn, no network.

Every axis is labelled and every chart states what "better" means, so the plots
cannot be read as showing more than they do. Charts with no data are skipped
rather than rendered empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: works on Kaggle and in CI
import matplotlib.pyplot as plt  # noqa: E402

BASE_COLOR = "#8c8c8c"
FT_COLOR = "#2b6cb0"


def _save(fig: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_loss_curves(history: list[dict[str, Any]], path: str | Path) -> str | None:
    """Training and validation loss against optimizer step, on shared axes."""
    train = [(h["step"], h["train_loss"]) for h in history if h.get("train_loss") is not None]
    val = [(h["step"], h["validation_loss"]) for h in history if h.get("validation_loss") is not None]
    if not train and not val:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    if train:
        ax.plot(*zip(*train, strict=True), label="Training loss", color=FT_COLOR, marker="o", markersize=3)
    if val:
        ax.plot(*zip(*val, strict=True), label="Validation loss", color="#c05621", marker="s", markersize=4)
        best_step, best_loss = min(val, key=lambda p: p[1])
        ax.axvline(best_step, color="#718096", linestyle="--", linewidth=1)
        ax.annotate(
            f"best val {best_loss:.3f}\n@ step {best_step}",
            xy=(best_step, best_loss),
            xytext=(6, 12),
            textcoords="offset points",
            fontsize=9,
            color="#4a5568",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (nats/token, lower is better)")
    ax.set_title("Training vs validation loss")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, Path(path))


def plot_gap(history: list[dict[str, Any]], path: str | Path) -> str | None:
    """Validation-minus-training loss over time: the overfitting signal."""
    by_step = {h["step"]: h for h in history}
    points = [
        (step, h["validation_loss"] - h["train_loss"])
        for step, h in sorted(by_step.items())
        if h.get("validation_loss") is not None and h.get("train_loss") is not None
    ]
    if len(points) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(*zip(*points, strict=True), color="#805ad5", marker="o", markersize=4)
    ax.axhline(0, color="#a0aec0", linewidth=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss - training loss")
    ax.set_title("Generalization gap over training\n(rising = the model is starting to memorise)")
    ax.grid(alpha=0.25)
    return _save(fig, Path(path))


def _grouped_bars(
    categories: list[str],
    base_values: list[float],
    ft_values: list[float],
    *,
    ylabel: str,
    title: str,
    path: str | Path,
) -> str:
    x = range(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(categories)), 5))
    bars_base = ax.bar([i - width / 2 for i in x], base_values, width, label="Base", color=BASE_COLOR)
    bars_ft = ax.bar([i + width / 2 for i in x], ft_values, width, label="Fine-tuned", color=FT_COLOR)
    for bars in (bars_base, bars_ft):
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, Path(path))


def plot_perplexity(perplexity: dict[str, Any], path: str | Path) -> str | None:
    splits = [s for s in ("train", "validation", "test") if s in perplexity.get("base", {})]
    if not splits:
        return None
    base = [float(perplexity["base"][s]) for s in splits]
    ft = [float(perplexity["fine_tuned"].get(s, 0.0)) for s in splits]
    if any(v == float("inf") for v in base + ft):
        return None
    return _grouped_bars(
        splits, base, ft,
        ylabel="Perplexity (lower is better)",
        title="Perplexity: base vs fine-tuned",
        path=path,
    )


def _comparison_bars(comparison: dict[str, Any], title: str, path: str | Path) -> str | None:
    if not comparison:
        return None
    metrics = ["exact_match", "normalized_exact_match", "token_f1", "contains_reference"]
    base = comparison.get("base", {})
    ft = comparison.get("fine_tuned", {})
    metrics = [m for m in metrics if m in base and m in ft]
    if not metrics:
        return None
    return _grouped_bars(
        metrics,
        [float(base[m]) for m in metrics],
        [float(ft[m]) for m in metrics],
        ylabel="Score, 0-1 (higher is better)",
        title=title,
        path=path,
    )


def plot_task_scores(task: dict[str, Any], path: str | Path) -> str | None:
    return _comparison_bars(task, "Task performance on held-out test set", path)


def plot_generalization(generalization: dict[str, Any], path: str | Path) -> str | None:
    return _comparison_bars(
        generalization, "Generalization: same facts, reworded questions", path
    )


def plot_forgetting(forgetting: dict[str, Any], path: str | Path) -> str | None:
    """Per-category general-capability comparison."""
    if not forgetting:
        return None
    metric = forgetting.get("primary_metric", "contains_reference")
    base_cats = forgetting.get("base", {}).get("by_category", {})
    ft_cats = forgetting.get("fine_tuned", {}).get("by_category", {})
    categories = sorted(base_cats)
    if not categories:
        return None
    return _grouped_bars(
        categories + ["overall"],
        [float(base_cats[c][metric]) for c in categories] + [float(forgetting["base"][metric])],
        [float(ft_cats.get(c, {}).get(metric, 0.0)) for c in categories]
        + [float(forgetting["fine_tuned"][metric])],
        ylabel=f"{metric}, 0-1 (higher is better)",
        title="Catastrophic forgetting: general-capability probe\n(bars dropping = capability lost)",
        path=path,
    )


def generate_all_plots(
    results: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    plots_dir: str | Path = "artifacts/evaluation/plots",
    training_dir: str | Path = "artifacts/training",
) -> dict[str, str]:
    """Render every plot that has data. Returns {name: path} for those written."""
    plots_dir = Path(plots_dir)
    written: dict[str, str | None] = {
        "training_vs_validation_loss": plot_loss_curves(
            history, plots_dir / "training_vs_validation_loss.png"
        ),
        "train_validation_gap": plot_gap(history, plots_dir / "train_validation_gap.png"),
        "perplexity_comparison": plot_perplexity(
            results.get("perplexity", {}), plots_dir / "perplexity_comparison.png"
        ),
        "base_vs_finetuned_task_score": plot_task_scores(
            results.get("task", {}), plots_dir / "base_vs_finetuned_task_score.png"
        ),
        "generalization_comparison": plot_generalization(
            results.get("generalization", {}), plots_dir / "generalization_comparison.png"
        ),
        "catastrophic_forgetting_comparison": plot_forgetting(
            results.get("forgetting", {}), plots_dir / "catastrophic_forgetting_comparison.png"
        ),
    }
    # The loss curve is also written next to the training artifacts, as specified.
    if history:
        plot_loss_curves(history, Path(training_dir) / "loss_curve.png")
        written["loss_curve"] = str(Path(training_dir) / "loss_curve.png")
    return {k: v for k, v in written.items() if v}
