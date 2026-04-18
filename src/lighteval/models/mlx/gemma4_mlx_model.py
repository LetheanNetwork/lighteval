from __future__ import annotations

import platform
from typing import List, Optional

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.models.transformers.gemma4_model import GenerationConfig
from lighteval.utils.cache_management import SampleCache


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class Gemma4MLXModel(LightevalModel):
    """LightevalModel wrapping an MLX checkpoint via `mlx-lm`.

    Expects an MLX-compatible repo id such as `mlx-community/gemma-4-e2b-it-bf16`
    or a local path produced by `mlx_lm.convert`. Apple Silicon only.
    """

    def __init__(
        self,
        model_path: str,
        generation: Optional[GenerationConfig] = None,
    ):
        if not is_apple_silicon():
            raise RuntimeError("Gemma4MLXModel requires Apple Silicon (arm64 Darwin).")

        try:
            from mlx_lm import load
        except ImportError as err:
            raise ImportError(
                "mlx-lm is required. Install with `pip install mlx-lm`."
            ) from err

        self.model_path = model_path
        self._gen = generation or GenerationConfig()

        self._model, self._tokenizer = load(model_path)

        self.config = ModelConfig(model_name=str(model_path))
        self._cache = SampleCache(self.config)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_length(self) -> int:
        cfg = getattr(self._model, "config", None) or getattr(self._model, "args", None)
        if cfg is not None:
            for attr in ("max_position_embeddings", "max_seq_len", "context_length"):
                value = getattr(cfg, attr, None)
                if value:
                    return int(value)
        return 131072

    @property
    def add_special_tokens(self) -> bool:
        return False

    def _build_sampler(self):
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=self._gen.temperature,
            top_p=self._gen.top_p,
            top_k=self._gen.top_k,
        )

    def greedy_until(self, docs, **kwargs) -> List[ModelResponse]:
        from tqdm.auto import tqdm

        try:
            from mlx_lm import generate
        except ImportError as err:
            raise ImportError("mlx-lm is required for inference.") from err

        sampler = self._build_sampler()
        responses = []
        for doc in tqdm(docs, desc="Gemma4MLXModel.greedy_until"):
            messages = [{"role": "user", "content": doc.query}]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._gen.enable_thinking,
            )
            text = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._gen.max_new_tokens,
                sampler=sampler,
                verbose=False,
            )
            responses.append(ModelResponse(text=[text]))
        return responses

    def loglikelihood(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemma4MLXModel does not implement loglikelihood — "
            "pick a task that uses greedy_until."
        )

    def loglikelihood_rolling(self, *args, **kwargs):
        raise NotImplementedError("Gemma4MLXModel does not implement loglikelihood_rolling.")

    def __repr__(self) -> str:
        return f"Gemma4MLXModel(model_path={self.model_path!r})"
