"""Single-prompt and interactive inference against the fine-tuned adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from src.inference.types import GenerationResult
from src.training.config import Config
from src.training.model import load_base_model, load_finetuned_model, load_tokenizer

DEFAULT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def is_adapter_reference(adapter_path: str | Path) -> bool:
    """True when this points at a loadable adapter: a local dir or a Hub repo id.

    Without the Hub case, passing "user/my-adapter" would look like a missing
    local path and the base model would be served instead - silently producing
    results that have nothing to do with the fine-tune.
    """
    path = Path(adapter_path)
    if path.exists():
        return True
    text = str(adapter_path)
    # Hub ids are "owner/name"; reject anything that looks like a filesystem path.
    return (
        text.count("/") == 1
        and not text.startswith((".", "/", "~"))
        and all(part.strip() for part in text.split("/"))
    )


class Generator:
    """Loads the base model (plus an optional LoRA adapter) once and reuses it."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | Path | None = None,
        *,
        quantization: dict[str, Any] | None = None,
        template: str = DEFAULT_TEMPLATE,
    ) -> None:
        self.template = template
        self.tokenizer = load_tokenizer(model_name)
        self.tokenizer.padding_side = "left"
        if adapter_path and is_adapter_reference(adapter_path):
            # Either a local directory or a Hugging Face Hub repo id - PEFT accepts both.
            self.model = load_finetuned_model(model_name, str(adapter_path), quantization)
            self.adapter_path = str(adapter_path)
        else:
            if adapter_path:
                print(
                    f"[warn] adapter not found at {adapter_path}; serving the BASE model. "
                    "Results below are NOT from your fine-tune."
                )
            self.model = load_base_model(model_name, quantization, for_training=False)
            self.adapter_path = None
        self.model.eval()

    @classmethod
    def from_config(cls, config: Config, adapter_path: str | Path | None) -> Generator:
        return cls(
            config.model_name,
            adapter_path,
            quantization=config.quantization,
            template=config.prompt["template"],
        )

    @torch.no_grad()
    def generate(
        self,
        instruction: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Greedy by default (temperature 0) so answers are reproducible."""
        prompt = self.template.format(instruction=instruction, response="")
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )

        sample = temperature > 0
        start = time.perf_counter()
        output = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature if sample else None,
            top_p=top_p if sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        latency = time.perf_counter() - start

        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(text, latency, int(new_tokens.shape[0]))


def interactive(generator: Generator, *, max_new_tokens: int = 256) -> None:
    """Simple REPL. Ctrl-D or 'exit' to quit."""
    banner = "adapter: " + (generator.adapter_path or "none (base model)")
    print(f"Interactive mode - {banner}. Type 'exit' to quit.\n")
    while True:
        try:
            prompt = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return
        result = generator.generate(prompt, max_new_tokens=max_new_tokens)
        print(f"model > {result.response}")
        print(
            f"        [{result.latency_seconds:.2f}s, {result.generated_tokens} tokens, "
            f"{result.tokens_per_second} tok/s]\n"
        )
