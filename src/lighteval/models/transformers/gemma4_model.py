# Copyright 2025 LetheanNetwork
# Licensed under the MIT License (see LICENSE)
#
# Gemma 4 support for lighteval's transformers backend.
#
# This module provides Gemma4Model, a LightevalModel subclass tuned for
# Google's Gemma 4 multimodal checkpoints. It uses AutoProcessor (the
# Gemma-4-correct loader) instead of AutoTokenizer, and handles the
# Gemma 4 chat template (including the "thinking" mode toggle) natively.
#
# Typical usage from a notebook:
#
#     from lighteval.models.transformers import Gemma4Model, GenerationConfig
#     model = Gemma4Model('/kaggle/input/.../gemma-4-e2b-it/1')
#     # pass to Pipeline(..., model=model) or use via lighteval.kaggle.Gemma4Eval
#
# Kaggle users: see lighteval.kaggle.Gemma4Eval for the notebook-first API
# that handles KaggleHub model paths, dual-GPU paired runs, and the dashboard.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.utils.cache_management import SampleCache


@dataclass
class GenerationConfig:
    """Sampling and generation knobs for Gemma 4.

    Defaults track Google's calibrated Gemma 4 sampling recipe
    (temperature=1.0, top_p=0.95, top_k=64). MCQ-style evals typically
    want enable_thinking=False so the model emits a direct answer.
    """

    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    enable_thinking: bool = False
    multi_turn: bool = False  # used by the Flax ChatSampler sibling


def _pick_dtype(dtype: str = "auto"):
    """Resolve a torch dtype. 'auto' = bf16 on Ampere+, fp16 on T4/V100, fp32 on CPU."""
    if dtype != "auto":
        return getattr(torch, dtype)
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


class Gemma4Model(LightevalModel):
    """LightevalModel wrapping a Gemma 4 checkpoint via transformers.

    Uses AutoProcessor (not AutoTokenizer) — this is the loader Google's
    Kaggle examples use and the one that correctly understands Gemma 4's
    multimodal chat template. We only exercise the text path here; the
    processor's tokenizer satisfies lighteval's abstract interface.

    Parameters
    ----------
    model_path : str
        Local directory or HuggingFace Hub repo id for a Gemma 4 checkpoint.
    device_map : str, default 'auto'
        Passed to AutoModelForCausalLM.from_pretrained.
    dtype : str, default 'auto'
        'auto' picks bf16 on Ampere+, fp16 on T4/V100, fp32 on CPU. Or pass
        an explicit dtype name: 'bfloat16', 'float16', 'float32'.
    generation : GenerationConfig | None
        Sampling config. If None, Gemma 4 calibrated defaults are used.

    Example
    -------
    >>> model = Gemma4Model('/kaggle/input/.../gemma-4-e2b-it/1')
    >>> model.max_length
    131072
    >>> # Pass into lighteval.pipeline.Pipeline(..., model=model)
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: str = "auto",
        generation: Optional[GenerationConfig] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_path = model_path
        self._gen = generation or GenerationConfig()

        self.processor = AutoProcessor.from_pretrained(model_path)
        tok = self.processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=_pick_dtype(dtype),
            device_map=device_map,
        )
        self.model.train(False)  # inference mode
        self.device = next(self.model.parameters()).device

        self.config = ModelConfig(model_name=str(model_path))
        self._cache = SampleCache(self.config)

    @classmethod
    def from_kagglehub(
        cls,
        kagglehub_slug: str,
        **kwargs,
    ) -> "Gemma4Model":
        """Construct a Gemma4Model from a KaggleHub model slug.

        Example
        -------
        >>> model = Gemma4Model.from_kagglehub(
        ...     'google/gemma-4/transformers/gemma-4-e2b-it'
        ... )
        """
        import kagglehub  # type: ignore

        path = kagglehub.model_download(kagglehub_slug)
        return cls(model_path=path, **kwargs)

    # --- Abstract properties required by LightevalModel ---
    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def max_length(self) -> int:
        return getattr(self.model.config, "max_position_embeddings", 8192)

    @property
    def add_special_tokens(self) -> bool:
        # The chat template adds them; lighteval's tok_encode must not double-add.
        return False

    # --- Inference entrypoint ---
    def greedy_until(self, docs, **kwargs) -> List[ModelResponse]:
        responses = []
        for doc in docs:
            messages = [{"role": "user", "content": doc.query}]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._gen.enable_thinking,
            )
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self._gen.max_new_tokens,
                    do_sample=True,
                    temperature=self._gen.temperature,
                    top_p=self._gen.top_p,
                    top_k=self._gen.top_k,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            gen = self.processor.decode(out[0][input_len:], skip_special_tokens=True)
            responses.append(ModelResponse(text=[gen]))
        return responses

    def loglikelihood(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemma4Model does not implement loglikelihood. "
            "Pick a task that uses greedy_until (MMLU-Pro, IFEval, etc.) "
            "or subclass Gemma4Model and add loglikelihood support."
        )

    def loglikelihood_rolling(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemma4Model does not implement loglikelihood_rolling."
        )

    def __repr__(self) -> str:
        return (
            f"Gemma4Model(model_path={self.model_path!r}, "
            f"device={self.device}, dtype={next(self.model.parameters()).dtype})"
        )
