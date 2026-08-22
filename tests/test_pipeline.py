"""End-to-end coverage for the CPU-only pipeline: reporting, orchestration, plots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.prepare import format_prompt, prepare, write_json
from src.data.schema import Record, write_jsonl
from src.data.validate import (
    build_dataset_report,
    dataset_hash,
    render_dataset_report,
    validate_dataset,
)
from src.evaluation.plots import (
    generate_all_plots,
    plot_gap,
    plot_loss_curves,
    plot_perplexity,
)


@pytest.fixture
def dataset_file(tmp_path, sample_records) -> Path:
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, sample_records, include_meta=False)
    return path


# ------------------------------------------------------------------ reporting


def test_dataset_report_counts(dataset_file, sample_records):
    records, report = validate_dataset(dataset_file)
    assert report["valid_records"] == len(sample_records)
    assert report["invalid_records"] == 0
    assert report["duplicate_records"] == 1  # the planted exact duplicate
    assert report["estimated_tokens"] > 0
    assert len(report["dataset_sha256"]) == 64


def test_dataset_report_includes_invalid_lines(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"instruction": "a", "response": "1"}\nBROKEN\n', encoding="utf-8")
    _, report = validate_dataset(path)
    assert report["valid_records"] == 1
    assert report["invalid_records"] == 1
    assert report["issues_by_reason"] == {"invalid_json": 1}


def test_length_statistics_are_sane(sample_records):
    report = build_dataset_report(sample_records, [], "test")
    chars = report["instruction_statistics"]["characters"]
    assert chars["min"] <= chars["median"] <= chars["max"]
    assert chars["count"] == len(sample_records)


def test_dataset_hash_is_order_independent(sample_records):
    assert dataset_hash(sample_records) == dataset_hash(list(reversed(sample_records)))


def test_dataset_hash_changes_with_content(sample_records):
    mutated = [*sample_records[:-1], Record("different question", "different answer", 99)]
    assert dataset_hash(sample_records) != dataset_hash(mutated)


def test_rendered_report_is_markdown_and_leaks_nothing(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"instruction": "TOP SECRET", "response": ""}\n', encoding="utf-8")
    _, report = validate_dataset(path)
    rendered = render_dataset_report(report)
    assert rendered.startswith("# Dataset Quality Report")
    assert "| Valid records | 0 |" in rendered
    assert "TOP SECRET" not in rendered


# --------------------------------------------------------------------- prompt


def test_format_prompt_training_and_inference():
    record = Record("Who?", "Ada", 0)
    template = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    assert format_prompt(record, template).endswith("Ada")
    assert format_prompt(record, template, for_inference=True).endswith("### Response:\n")


# ----------------------------------------------------------------- pipeline


def test_prepare_writes_every_artifact(dataset_file, tmp_path):
    processed, reports = tmp_path / "processed", tmp_path / "reports"
    summary = prepare(
        dataset_file, threshold=0.8, seed=42, processed_dir=processed, report_dir=reports
    )

    for name in ("train", "validation", "test"):
        assert (processed / f"{name}.jsonl").exists()
    for name in ("dataset_report.json", "duplicate_report.json", "leakage_report.json"):
        assert json.loads((reports / name).read_text())
    assert (reports / "dataset_report.md").read_text().startswith("#")
    assert summary["leakage_report"]["leakage_free"]


def test_prepare_is_reproducible(dataset_file, tmp_path):
    kwargs = dict(threshold=0.8, seed=42, write_outputs=False)
    first = prepare(dataset_file, **kwargs)
    second = prepare(dataset_file, **kwargs)
    for name in ("train", "validation", "test"):
        assert [r.id for r in first["splits"][name]] == [r.id for r in second["splits"][name]]


def test_prepare_rejects_an_all_invalid_dataset(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text("NOT JSON\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="No valid records"):
        prepare(path, write_outputs=False)


def test_write_json_creates_parent_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c.json"
    write_json(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}


# ------------------------------------------------------------------- plots


def test_loss_curve_is_written(healthy_history, tmp_path):
    path = plot_loss_curves(healthy_history, tmp_path / "loss.png")
    assert path and Path(path).stat().st_size > 0


def test_loss_curve_skipped_when_there_is_no_data(tmp_path):
    assert plot_loss_curves([], tmp_path / "loss.png") is None


def test_gap_plot_needs_two_paired_points(tmp_path):
    assert plot_gap([{"step": 1, "train_loss": 1.0}], tmp_path / "gap.png") is None


def test_perplexity_plot_skips_infinite_values(tmp_path):
    diverged = {"base": {"test": 10.0}, "fine_tuned": {"test": float("inf")}}
    assert plot_perplexity(diverged, tmp_path / "ppl.png") is None


def test_generate_all_plots_writes_what_it_can(healthy_history, tmp_path):
    results = {
        "perplexity": {
            "base": {"train": 14.0, "validation": 15.0, "test": 15.4},
            "fine_tuned": {"train": 3.1, "validation": 5.2, "test": 5.6},
        },
        "task": {
            "base": {"exact_match": 0.0, "normalized_exact_match": 0.05,
                     "token_f1": 0.2, "contains_reference": 0.1},
            "fine_tuned": {"exact_match": 0.3, "normalized_exact_match": 0.42,
                           "token_f1": 0.68, "contains_reference": 0.55},
        },
        "forgetting": {
            "primary_metric": "contains_reference",
            "base": {"contains_reference": 0.6, "by_category": {"math": {"contains_reference": 0.6}}},
            "fine_tuned": {"contains_reference": 0.5, "by_category": {"math": {"contains_reference": 0.5}}},
        },
    }
    written = generate_all_plots(
        results, healthy_history, plots_dir=tmp_path / "plots", training_dir=tmp_path
    )
    for name in (
        "training_vs_validation_loss",
        "train_validation_gap",
        "perplexity_comparison",
        "base_vs_finetuned_task_score",
        "catastrophic_forgetting_comparison",
    ):
        assert name in written, f"{name} should have been plotted"
        assert Path(written[name]).exists()
    # No generalization data was supplied, so that plot must be absent, not empty.
    assert "generalization_comparison" not in written


def test_generate_all_plots_with_no_data_writes_nothing(tmp_path):
    assert generate_all_plots({}, [], plots_dir=tmp_path / "p", training_dir=tmp_path) == {}
