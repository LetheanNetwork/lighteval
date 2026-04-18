from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.utils.cache_management import SampleCache


@dataclass
class GenerationConfig:
    """Sampling knobs for Gemma 4 — defaults match Google's calibrated recipe."""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    enable_thinking: bool = False
    multi_turn: bool = False


def _pick_dtype(dtype: str = "auto"):
    if dtype != "auto":
        return getattr(torch, dtype)
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


class Gemma4Model(LightevalModel):
    """LightevalModel wrapping a Gemma 4 checkpoint loaded via `transformers`.

    Uses AutoProcessor rather than AutoTokenizer so the Gemma 4 chat template
    and its optional thinking-mode toggle are applied correctly.
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
        self.model.train(False)
        self.device = next(self.model.parameters()).device

        self.config = ModelConfig(model_name=str(model_path))
        self._cache = SampleCache(self.config)

    @classmethod
    def from_kagglehub(
        cls,
        kagglehub_slug: str,
        **kwargs,
    ) -> "Gemma4Model":
        """Download a checkpoint from KaggleHub and instantiate a Gemma4Model."""
        import kagglehub  # type: ignore

        path = kagglehub.model_download(kagglehub_slug)
        return cls(model_path=path, **kwargs)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def max_length(self) -> int:
        return getattr(self.model.config, "max_position_embeddings", 8192)

    @property
    def add_special_tokens(self) -> bool:
        return False

    def greedy_until(self, docs, **kwargs) -> List[ModelResponse]:
        from tqdm.auto import tqdm

        responses = []
        for i, doc in enumerate(tqdm(docs, desc="Gemma4Model.greedy_until")):
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

            del inputs, out
            if (i + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        return responses

    def loglikelihood(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemma4Model does not implement loglikelihood — "
            "pick a task that uses greedy_until (MMLU-Pro, IFEval, etc.)."
        )

    def loglikelihood_rolling(self, *args, **kwargs):
        raise NotImplementedError("Gemma4Model does not implement loglikelihood_rolling.")

    def __repr__(self) -> str:
        return (
            f"Gemma4Model(model_path={self.model_path!r}, "
            f"device={self.device}, dtype={next(self.model.parameters()).dtype})"
        )
