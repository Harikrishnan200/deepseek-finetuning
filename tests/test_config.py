"""Config loading and validation, including the real config files in configs/."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.training.config import (
    Config,
    ConfigError,
    load_config,
    load_evaluation_config,
    validate_config,
)

CONFIG_DIR = Path("configs")


def base_config() -> dict:
    return yaml.safe_load(Path("configs/qlora.yaml").read_text())


def test_real_qlora_config_loads():
    config = load_config("configs/qlora.yaml")
    assert isinstance(config, Config)
    assert config.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert config.seed == 42
    assert config.effective_batch_size == (
        config.training["per_device_train_batch_size"]
        * config.training["gradient_accumulation_steps"]
    )


@pytest.mark.parametrize(
    "path", sorted(CONFIG_DIR.glob("**/*.yaml")), ids=lambda p: str(p)
)
def test_every_shipped_config_is_valid(path):
    if path.name == "evaluation.yaml":
        assert load_evaluation_config(path)["gate"]
    else:
        assert load_config(path).model_name


def test_experiment_configs_differ_from_baseline():
    baseline = load_config("configs/qlora.yaml")
    for path in sorted(CONFIG_DIR.glob("experiments/*.yaml")):
        experiment = load_config(path)
        assert experiment.run_name != baseline.run_name
        assert experiment.training["output_dir"] != baseline.training["output_dir"], (
            f"{path} would overwrite the baseline artifacts"
        )


def test_split_ratios_must_sum_to_one():
    data = base_config()
    data["split"]["test_ratio"] = 0.25
    with pytest.raises(ConfigError, match="sum to 1.0"):
        validate_config(data)


def test_missing_top_level_key_rejected():
    data = base_config()
    del data["lora"]
    with pytest.raises(ConfigError, match="lora"):
        validate_config(data)


def test_prompt_template_must_contain_placeholders():
    data = base_config()
    data["prompt"]["template"] = "### Instruction:\n{instruction}\n"
    with pytest.raises(ConfigError, match=r"\{response\}"):
        validate_config(data)


def test_response_marker_must_appear_in_template():
    data = base_config()
    data["prompt"]["response_marker"] = "### Answer:\n"
    with pytest.raises(ConfigError, match="response_marker"):
        validate_config(data)


@pytest.mark.parametrize("value", [0, -1, -0.5])
def test_non_positive_learning_rate_rejected(value):
    data = base_config()
    data["training"]["learning_rate"] = value
    with pytest.raises(ConfigError, match="learning_rate"):
        validate_config(data)


def test_zero_batch_size_rejected():
    data = base_config()
    data["training"]["per_device_train_batch_size"] = 0
    with pytest.raises(ConfigError, match="per_device_train_batch_size"):
        validate_config(data)


def test_bad_lora_rank_rejected():
    data = base_config()
    data["lora"]["r"] = 0
    with pytest.raises(ConfigError, match="lora.r"):
        validate_config(data)


def test_bad_lora_dropout_rejected():
    data = base_config()
    data["lora"]["dropout"] = 1.0
    with pytest.raises(ConfigError, match="dropout"):
        validate_config(data)


def test_bad_quant_type_rejected():
    data = base_config()
    data["quantization"]["quant_type"] = "int8"
    with pytest.raises(ConfigError, match="quant_type"):
        validate_config(data)


def test_bad_compute_dtype_rejected():
    data = base_config()
    data["quantization"]["compute_dtype"] = "float8"
    with pytest.raises(ConfigError, match="compute_dtype"):
        validate_config(data)


def test_bad_near_duplicate_threshold_rejected():
    data = base_config()
    data["split"]["near_duplicate_threshold"] = 1.5
    with pytest.raises(ConfigError, match="near_duplicate_threshold"):
        validate_config(data)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_config_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_evaluation_config_gate_keys_present():
    gate = load_evaluation_config("configs/evaluation.yaml")["gate"]
    for key in (
        "minimum_task_accuracy_improvement",
        "maximum_allowed_forgetting",
        "maximum_test_perplexity",
        "maximum_leakage_rate",
        "allow_overfitting",
    ):
        assert key in gate


def test_compute_dtype_never_picks_float16_on_cpu():
    """float16 matmul is unsupported/glacial on CPU; float32 is the safe default."""
    import torch

    from src.training.model import best_device, resolve_compute_dtype

    dtype = resolve_compute_dtype("auto")
    if best_device() == "cpu":
        assert dtype is torch.float32
    assert resolve_compute_dtype("bfloat16") is torch.bfloat16


def test_quantization_disabled_without_cuda():
    """4-bit is CUDA-only: fall back to unquantized rather than crashing."""
    import torch

    from src.training.model import build_quantization_config

    config = build_quantization_config({"load_in_4bit": True, "quant_type": "nf4"})
    if not torch.cuda.is_available():
        assert config is None
    assert build_quantization_config({"load_in_4bit": False}) is None
