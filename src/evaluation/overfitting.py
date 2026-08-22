"""Overfitting and underfitting analysis from the training history.

Both analyses are *heuristics over loss curves*. They are useful early-warning
signals, not proofs. In particular the underfitting detector cannot distinguish
"the model has not learned yet" from "this dataset is intrinsically hard", so it
reports a status plus the signals that produced it and expects a human to read them.
"""

from __future__ import annotations

from typing import Any

HEALTHY = "healthy"
POSSIBLE_OVERFITTING = "possible_overfitting"
STRONG_OVERFITTING = "strong_overfitting"
POSSIBLE_UNDERFITTING = "possible_underfitting"
INSUFFICIENT_DATA = "insufficient_data"

DEFAULT_OVERFITTING = {
    "possible_overfitting_gap": 0.30,
    "strong_overfitting_gap": 0.80,
    "val_loss_rise_ratio": 0.10,
}
DEFAULT_UNDERFITTING = {
    "high_loss_threshold": 2.00,
    "small_gap_threshold": 0.10,
    "min_relative_loss_improvement": 0.10,
}


def _series(history: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [h for h in history if h.get(key) is not None]


def _last(history: list[dict[str, Any]], key: str) -> float | None:
    points = _series(history, key)
    return float(points[-1][key]) if points else None


def analyse_overfitting(
    history: list[dict[str, Any]], thresholds: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Classify a run as healthy / possible_overfitting / strong_overfitting.

    Three signals are combined:

    1. the generalization gap (final validation loss - final training loss),
    2. how far the final validation loss has risen above its best value,
    3. whether training loss is still falling while validation loss rises.
    """
    thresholds = {**DEFAULT_OVERFITTING, **(thresholds or {})}
    val_points = _series(history, "validation_loss")
    train_points = _series(history, "train_loss")

    if len(val_points) < 2 or not train_points:
        return {
            "status": INSUFFICIENT_DATA,
            "reason": "need at least two validation points and one training point",
            "evaluation_points": len(val_points),
            "thresholds": thresholds,
        }

    best = min(val_points, key=lambda p: p["validation_loss"])
    final_val = float(val_points[-1]["validation_loss"])
    final_train = float(train_points[-1]["train_loss"])
    best_val = float(best["validation_loss"])
    gap = final_val - final_train

    rise_ratio = (final_val - best_val) / best_val if best_val > 0 else 0.0
    half = max(1, len(val_points) // 2)
    val_rising = final_val > float(val_points[-half]["validation_loss"])
    train_falling = final_train < float(train_points[max(0, len(train_points) - half)]["train_loss"])

    signals = {
        "generalization_gap": round(gap, 4),
        "validation_loss_rise_ratio": round(rise_ratio, 4),
        "validation_loss_rising": bool(val_rising),
        "training_loss_still_falling": bool(train_falling),
        "final_is_best_checkpoint": best is val_points[-1],
    }

    if gap >= thresholds["strong_overfitting_gap"] or (
        rise_ratio >= thresholds["val_loss_rise_ratio"] and val_rising and train_falling
    ):
        status = STRONG_OVERFITTING
    elif gap >= thresholds["possible_overfitting_gap"] or rise_ratio > 0:
        status = POSSIBLE_OVERFITTING
    else:
        status = HEALTHY

    return {
        "status": status,
        "overfitting_detected": status != HEALTHY,
        "best_step": best["step"],
        "best_epoch": best.get("epoch"),
        "best_validation_loss": round(best_val, 4),
        "final_training_loss": round(final_train, 4),
        "final_validation_loss": round(final_val, 4),
        "generalization_gap": round(gap, 4),
        "signals": signals,
        "thresholds": thresholds,
        "note": "Heuristic based on loss curves; read the signals alongside the status.",
    }


def analyse_underfitting(
    history: list[dict[str, Any]], thresholds: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Heuristic underfitting check.

    Flags a run when losses stay high, train and validation stay close, and the
    training loss barely improved from its starting value. This is explicitly a
    heuristic - it is NOT a mathematically sound classifier, and a genuinely hard
    dataset can trip it while the model is learning fine.
    """
    thresholds = {**DEFAULT_UNDERFITTING, **(thresholds or {})}
    train_points = _series(history, "train_loss")
    val_points = _series(history, "validation_loss")

    if len(train_points) < 2:
        return {
            "status": INSUFFICIENT_DATA,
            "reason": "need at least two training loss points",
            "thresholds": thresholds,
            "note": "Underfitting detection is heuristic, not a proof.",
        }

    first_train = float(train_points[0]["train_loss"])
    final_train = float(train_points[-1]["train_loss"])
    final_val = _last(val_points, "validation_loss")
    improvement = (first_train - final_train) / first_train if first_train > 0 else 0.0
    gap = (final_val - final_train) if final_val is not None else None

    signals = {
        "high_training_loss": final_train >= thresholds["high_loss_threshold"],
        "high_validation_loss": (
            final_val is not None and final_val >= thresholds["high_loss_threshold"]
        ),
        "small_generalization_gap": (
            gap is not None and abs(gap) <= thresholds["small_gap_threshold"]
        ),
        "insufficient_loss_improvement": improvement < thresholds["min_relative_loss_improvement"],
        "relative_loss_improvement": round(improvement, 4),
    }

    triggered = sum(
        1
        for key in (
            "high_training_loss",
            "high_validation_loss",
            "small_generalization_gap",
            "insufficient_loss_improvement",
        )
        if signals[key]
    )
    # Require the loss to actually be high, plus at least one corroborating signal.
    status = (
        POSSIBLE_UNDERFITTING
        if signals["high_training_loss"] and triggered >= 2
        else HEALTHY
    )

    return {
        "status": status,
        "underfitting_detected": status == POSSIBLE_UNDERFITTING,
        "first_training_loss": round(first_train, 4),
        "final_training_loss": round(final_train, 4),
        "final_validation_loss": round(final_val, 4) if final_val is not None else None,
        "generalization_gap": round(gap, 4) if gap is not None else None,
        "signals": signals,
        "triggered_signals": triggered,
        "thresholds": thresholds,
        "note": (
            "Heuristic only. High loss on a small, hard, or noisy dataset can look "
            "identical to underfitting. Compare against the base model before acting."
        ),
    }
