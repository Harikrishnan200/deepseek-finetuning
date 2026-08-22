"""Model, tokenizer, quantization and LoRA setup.

Everything torch-dependent lives here so the data and gate layers stay importable
in CPU-only CI.

LoRA target modules are **discovered from the loaded model**, not hard-coded.
DeepSeek-R1-Distill-Qwen-1.5B is a ``Qwen2ForCausalLM`` (28 layers, hidden 1536,
GQA with 2 KV heads), whose attention/MLP projections are named
``q_proj/k_proj/v_proj/o_proj`` and ``gate_proj/up_proj/down_proj`` - but the code
below verifies that against the real module tree and raises if it cannot find
them, rather than silently adapting nothing.
"""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Projection suffixes we look for, in preference order. Attention first: if the
# model turns out to use different names, the assertion in find_target_modules fires.
CANDIDATE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def best_device() -> str:
    """cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_compute_dtype(requested: str = "auto") -> torch.dtype:
    """Pick the compute dtype for the current device.

    On CUDA, prefer bfloat16 when supported (Kaggle's T4 is compute capability
    7.5 and is not, so it falls back to fp16). On Apple Silicon, bfloat16 works.
    On plain CPU, float16 matmuls are unsupported or crawl, so use float32.
    """
    if requested != "auto":
        return DTYPE_MAP[requested]
    device = best_device()
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def build_quantization_config(quantization: dict[str, Any]) -> BitsAndBytesConfig | None:
    """Build a 4-bit config, or None when quantization is off or unavailable.

    bitsandbytes 4-bit kernels are CUDA-only. Rather than crashing on a laptop,
    fall back to unquantized weights - a 1.5B model is ~3.5 GB in bf16, which is
    fine for local inference.
    """
    if not quantization.get("load_in_4bit", True):
        return None
    if not torch.cuda.is_available():
        print(
            "[warn] no CUDA GPU: bitsandbytes 4-bit is unavailable, loading "
            "unquantized weights instead (needs ~3.5 GB of RAM)."
        )
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=quantization.get("double_quant", True),
        bnb_4bit_compute_dtype=resolve_compute_dtype(quantization.get("compute_dtype", "auto")),
    )


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Causal LM training pads on the right; generation pads on the left.
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str,
    quantization: dict[str, Any] | None = None,
    *,
    for_training: bool = True,
) -> Any:
    """Load the base model, 4-bit quantized when a quantization config is given."""
    quant_config = build_quantization_config(quantization or {})
    dtype = resolve_compute_dtype((quantization or {}).get("compute_dtype", "auto"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if quant_config is None and not torch.cuda.is_available():
        # device_map is CUDA-only here, so place the model explicitly.
        model = model.to(best_device())
    model.config.use_cache = not for_training
    return model


def find_target_modules(model: Any) -> list[str]:
    """Inspect the real module tree and return the projection names to adapt.

    Any linear layer whose name ends with one of CANDIDATE_SUFFIXES is a target.
    ``lm_head`` is deliberately excluded - adapting the 152k-row output projection
    would dominate the adapter size for no benefit here.
    """
    found: set[str] = set()
    for name, module in model.named_modules():
        if not _is_linear(module):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in CANDIDATE_SUFFIXES and "lm_head" not in name:
            found.add(leaf)

    if not found:
        available = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules() if _is_linear(m)})
        raise RuntimeError(
            "Could not auto-detect LoRA target modules on "
            f"{type(model).__name__}. Linear leaf names present: {available}. "
            "Set lora.target_modules explicitly in the config."
        )
    # Keep the canonical order rather than set order, for reproducible configs.
    return [s for s in CANDIDATE_SUFFIXES if s in found]


def _is_linear(module: Any) -> bool:
    """True for nn.Linear and for bitsandbytes' quantized replacements."""
    if isinstance(module, torch.nn.Linear):
        return True
    return type(module).__name__ in {"Linear4bit", "Linear8bitLt", "Params4bit"}


def count_parameters(model: Any) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percentage": round(100.0 * trainable / total, 6) if total else 0.0,
    }


def build_lora_config(lora: dict[str, Any], target_modules: list[str]) -> LoraConfig:
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora.get("bias", "none"),
        task_type=lora.get("task_type", "CAUSAL_LM"),
        target_modules=target_modules,
    )


def prepare_peft_model(
    model: Any,
    lora: dict[str, Any],
    *,
    gradient_checkpointing: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Wrap a quantized base model with LoRA adapters.

    Returns the PEFT model and a parameter-efficiency summary.
    """
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    target_modules = list(lora.get("target_modules") or []) or find_target_modules(model)
    peft_model = get_peft_model(model, build_lora_config(lora, target_modules))

    summary = count_parameters(peft_model)
    summary["target_modules"] = target_modules
    summary["lora_r"] = lora["r"]
    summary["lora_alpha"] = lora["alpha"]
    summary["lora_dropout"] = lora["dropout"]
    return peft_model, summary


def load_finetuned_model(
    model_name: str, adapter_path: str, quantization: dict[str, Any] | None = None
) -> Any:
    """Load the base model with a trained LoRA adapter attached, for evaluation."""
    from peft import PeftModel

    base = load_base_model(model_name, quantization, for_training=False)
    return PeftModel.from_pretrained(base, adapter_path)
