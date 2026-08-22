"""Metrics, overfitting/underfitting heuristics, generalization, forgetting, and the gate."""

from __future__ import annotations

import math

import pytest

from src.data.schema import Record
from src.evaluation.forgetting import compare_forgetting, evaluate_general_knowledge
from src.evaluation.generalization import audit_generalization_set, compare_generalization
from src.evaluation.metrics import (
    aggregate_scores,
    contains_reference,
    evaluate_predictions,
    exact_match,
    normalize_answer,
    normalized_exact_match,
    perplexity_from_loss,
    score_predictions,
    token_f1,
)
from src.evaluation.overfitting import analyse_overfitting, analyse_underfitting
from src.evaluation.report import FAIL, PASS, PASS_WITH_WARNINGS, evaluate_gate, render_final_report

# --------------------------------------------------------------------- metrics


def test_normalize_answer_strips_case_punctuation_and_articles():
    assert normalize_answer("The  N. Harikrishnan!") == "n harikrishnan"


def test_exact_match_is_strict():
    assert exact_match("Ada Lovelace", "Ada Lovelace") == 1.0
    assert exact_match("ada lovelace", "Ada Lovelace") == 0.0


def test_normalized_exact_match_ignores_surface_form():
    assert normalized_exact_match("N Harikrishnan.", "n  harikrishnan") == 1.0
    assert normalized_exact_match("Someone else", "n harikrishnan") == 0.0


def test_token_f1_gives_partial_credit():
    score = token_f1("His name is N Harikrishnan", "N Harikrishnan")
    assert 0.0 < score < 1.0


def test_token_f1_is_one_for_identical_normalized_answers():
    assert token_f1("N Harikrishnan.", "n harikrishnan") == pytest.approx(1.0)


def test_token_f1_is_zero_with_no_overlap():
    assert token_f1("completely different", "n harikrishnan") == 0.0


def test_token_f1_handles_empty_strings():
    assert token_f1("", "") == 1.0
    assert token_f1("", "something") == 0.0


def test_contains_reference_matches_whole_tokens_only():
    assert contains_reference("The answer is 43.", "43") == 1.0
    assert contains_reference("The answer is 431.", "43") == 0.0
    assert contains_reference("It is in Paris, France.", "Paris") == 1.0


def test_score_predictions_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        score_predictions(["a"], ["a", "b"])


def test_aggregate_scores_on_empty_input_is_zero_not_nan():
    aggregated = aggregate_scores([])
    assert all(value == 0.0 for value in aggregated.values())


def test_evaluate_predictions_reports_count():
    result = evaluate_predictions(["Ada", "Bob"], ["Ada", "Ada"])
    assert result["count"] == 2
    assert result["normalized_exact_match"] == 0.5


def test_perplexity_from_loss():
    assert perplexity_from_loss(0.0) == 1.0
    assert perplexity_from_loss(math.log(2)) == pytest.approx(2.0, abs=1e-3)
    assert perplexity_from_loss(1000) == float("inf")
    assert math.isnan(perplexity_from_loss(float("nan")))


# ------------------------------------------------------ overfitting/underfitting


def test_healthy_run_is_classified_healthy(healthy_history):
    result = analyse_overfitting(healthy_history)
    assert result["status"] == "healthy"
    assert result["overfitting_detected"] is False


def test_overfitting_run_is_detected(overfitting_history):
    result = analyse_overfitting(overfitting_history)
    assert result["status"] in ("possible_overfitting", "strong_overfitting")
    assert result["overfitting_detected"] is True
    assert result["generalization_gap"] > 0


def test_overfitting_reports_the_best_checkpoint(overfitting_history):
    result = analyse_overfitting(overfitting_history)
    best = min(h["validation_loss"] for h in overfitting_history)
    assert result["best_validation_loss"] == pytest.approx(best, abs=1e-3)
    assert result["best_step"] < overfitting_history[-1]["step"]


def test_overfitting_thresholds_are_configurable(healthy_history):
    strict = analyse_overfitting(
        healthy_history, {"possible_overfitting_gap": 0.0001, "strong_overfitting_gap": 0.001}
    )
    assert strict["status"] != "healthy"


def test_overfitting_needs_enough_data():
    assert analyse_overfitting([])["status"] == "insufficient_data"
    assert analyse_overfitting([{"step": 1, "train_loss": 1.0}])["status"] == "insufficient_data"


def test_underfitting_run_is_detected(underfitting_history):
    result = analyse_underfitting(underfitting_history)
    assert result["status"] == "possible_underfitting"
    assert result["signals"]["high_training_loss"] is True
    assert "heuristic" in result["note"].lower()


def test_healthy_run_is_not_flagged_as_underfitting(healthy_history):
    assert analyse_underfitting(healthy_history)["status"] == "healthy"


def test_underfitting_needs_enough_data():
    assert analyse_underfitting([])["status"] == "insufficient_data"


# ---------------------------------------------- generalization and forgetting


def _stub(answers):
    return lambda instructions: list(answers)[: len(instructions)]


def test_generalization_comparison_reports_improvement():
    records = [Record("Complete name?", "Ada Lovelace", 0), Record("Birth city?", "London", 1)]
    result = compare_generalization(
        records, _stub(["no idea", "somewhere"]), _stub(["Ada Lovelace", "London"])
    )
    assert result["fine_tuned"]["token_f1"] > result["base"]["token_f1"]
    assert result["primary_improvement"] > 0


def test_generalization_audit_flags_near_copies(sample_records):
    """TF-IDF needs a realistic corpus, so the audit runs against a full train set."""
    train = sample_records
    leaky = [Record("What was Ada's full name?", "Ada Lovelace", 0)]
    clean = [Record("Which mathematician mentored her?", "Charles Babbage", 0)]
    assert audit_generalization_set(leaky, train, threshold=0.80)["valid"] is False
    assert audit_generalization_set(clean, train, threshold=0.80)["valid"] is True


def test_real_generalization_set_has_no_training_near_copies():
    from pathlib import Path

    from src.data.schema import load_records

    gen_path, train_path = Path("data/eval/generalization.jsonl"), Path("data/processed/train.jsonl")
    if not (gen_path.exists() and train_path.exists()):
        pytest.skip("generalization set or training split not present")
    gen, _ = load_records(gen_path)
    train, _ = load_records(train_path)
    audit = audit_generalization_set(gen, train, threshold=0.90)
    assert audit["valid"], audit["too_similar"]


def test_forgetting_is_positive_when_capability_drops():
    records = [
        Record("2+2?", "4", 0, category="math"),
        Record("Capital of France?", "Paris", 1, category="factual"),
    ]
    result = compare_forgetting(
        records, _stub(["4", "Paris"]), _stub(["I do not know", "N Harikrishnan"])
    )
    assert result["forgetting"] == pytest.approx(1.0)
    assert result["within_tolerance"] is False


def test_forgetting_is_zero_when_capability_is_preserved():
    records = [Record("2+2?", "4", 0, category="math")]
    answers = ["The answer is 4."]
    result = compare_forgetting(records, _stub(answers), _stub(answers))
    assert result["forgetting"] == 0.0
    assert result["within_tolerance"] is True


def test_general_knowledge_scores_break_down_by_category():
    records = [
        Record("2+2?", "4", 0, category="math"),
        Record("Capital of France?", "Paris", 1, category="factual"),
    ]
    result = evaluate_general_knowledge(records, _stub(["4", "Berlin"]))
    assert result["by_category"]["math"]["contains_reference"] == 1.0
    assert result["by_category"]["factual"]["contains_reference"] == 0.0


# ------------------------------------------------------------------- the gate


def passing_results() -> dict:
    return {
        "task": {"improvement": {"normalized_exact_match": 0.30}},
        "generalization": {"primary_improvement": 0.20},
        "forgetting": {"forgetting": 0.02},
        "perplexity": {"fine_tuned": {"test": 4.2}},
        "leakage": {"max_overlap_rate": 0.0},
        "overfitting": {"status": "healthy"},
        "underfitting": {"status": "healthy"},
    }


def test_gate_passes_when_everything_is_good():
    result = evaluate_gate(passing_results())
    assert result["verdict"] == PASS
    assert result["promote"] is True
    assert not result["failed_checks"]


def test_gate_fails_on_insufficient_task_improvement():
    results = passing_results()
    results["task"]["improvement"]["normalized_exact_match"] = 0.01
    result = evaluate_gate(results)
    assert result["verdict"] == FAIL
    assert "task_accuracy_improvement" in result["failed_checks"]


def test_gate_fails_on_excess_forgetting():
    results = passing_results()
    results["forgetting"]["forgetting"] = 0.35
    assert "catastrophic_forgetting" in evaluate_gate(results)["failed_checks"]


def test_gate_fails_on_leakage():
    results = passing_results()
    results["leakage"]["max_overlap_rate"] = 0.02
    assert "data_leakage" in evaluate_gate(results)["failed_checks"]


def test_gate_fails_on_high_perplexity():
    results = passing_results()
    results["perplexity"]["fine_tuned"]["test"] = 500.0
    assert "test_perplexity" in evaluate_gate(results)["failed_checks"]


def test_gate_fails_on_strong_overfitting():
    results = passing_results()
    results["overfitting"]["status"] = "strong_overfitting"
    assert "overfitting" in evaluate_gate(results)["failed_checks"]


def test_gate_allows_overfitting_when_configured():
    results = passing_results()
    results["overfitting"]["status"] = "strong_overfitting"
    result = evaluate_gate(results, {"allow_overfitting": True})
    assert "overfitting" not in result["failed_checks"]


def test_gate_warns_on_possible_overfitting():
    results = passing_results()
    results["overfitting"]["status"] = "possible_overfitting"
    result = evaluate_gate(results)
    assert result["verdict"] == PASS_WITH_WARNINGS
    assert result["promote"] is False


def test_gate_warns_on_possible_underfitting():
    results = passing_results()
    results["underfitting"]["status"] = "possible_underfitting"
    assert evaluate_gate(results)["verdict"] == PASS_WITH_WARNINGS


def test_missing_measurement_becomes_a_warning_not_a_silent_pass():
    results = passing_results()
    del results["forgetting"]
    result = evaluate_gate(results)
    assert result["verdict"] == PASS_WITH_WARNINGS
    assert any("forgetting" in w for w in result["warnings"])


def test_gate_fails_when_nothing_can_be_evaluated():
    result = evaluate_gate({})
    assert result["verdict"] == FAIL
    assert result["promote"] is False


def test_gate_thresholds_come_from_config():
    results = passing_results()
    results["task"]["improvement"]["normalized_exact_match"] = 0.10
    assert evaluate_gate(results, {"minimum_task_accuracy_improvement": 0.05})["verdict"] == PASS
    assert evaluate_gate(results, {"minimum_task_accuracy_improvement": 0.50})["verdict"] == FAIL


# ----------------------------------------------------------------- the report


def test_final_report_renders_all_sections(healthy_history):
    results = passing_results()
    results["overfitting"] = analyse_overfitting(healthy_history)
    results["underfitting"] = analyse_underfitting(healthy_history)
    results["dataset"] = {"total": 889, "train": 622, "validation": 134, "test": 133}
    results["metadata"] = {"run_name": "t", "model_name": "m", "seed": 42}
    results["task"] = {
        "base": {"normalized_exact_match": 0.05, "token_f1": 0.2},
        "fine_tuned": {"normalized_exact_match": 0.42, "token_f1": 0.7},
        "improvement": {"normalized_exact_match": 0.37, "token_f1": 0.5},
    }
    results["gate"] = evaluate_gate(results)

    report = render_final_report(results)
    for heading in (
        "# Model Fine-Tuning Evaluation Report",
        "## Dataset",
        "## Training",
        "## Training Behavior",
        "## Perplexity",
        "## Task Performance",
        "## Generalization",
        "## Catastrophic Forgetting",
        "## Leakage",
        "## Final Recommendation",
    ):
        assert heading in report
    assert results["gate"]["verdict"] in report


def test_report_survives_a_mostly_empty_results_dict():
    report = render_final_report({})
    assert "# Model Fine-Tuning Evaluation Report" in report
    assert "FAIL" in report


def test_report_describes_quantization_honestly():
    from src.evaluation.report import _describe_quantization

    assert _describe_quantization({"load_in_4bit": True, "quant_type": "nf4", "double_quant": True}) == (
        "4-bit nf4, double_quant=True"
    )
    # A config with 4-bit disabled must not be reported as 4-bit.
    assert "disabled" in _describe_quantization({"load_in_4bit": False, "quant_type": "nf4"})
    assert _describe_quantization({}) == "n/a"
