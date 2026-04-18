from __future__ import annotations

from typing import List, Optional

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.models.transformers.gemma4_model import GenerationConfig
from lighteval.utils.cache_management import SampleCache


class Gemma4FlaxModel(LightevalModel):
    """LightevalModel wrapping the reference Flax Gemma 4 implementation (`gemma.gm`).

    Requires the optional `gemma` library (install from
    https://github.com/google-deepmind/gemma).
    """

    def __init__(
        self,
        params_path: str,
        model_class_name: str = "Gemma4_E2B",
        generation: Optional[GenerationConfig] = None,
    ):
        from gemma import gm

        self.model_path = params_path
        self.model_class_name = model_class_name
        self._gen = generation or GenerationConfig()

        cls = self._resolve_model_class(gm, model_class_name)
        self._model = cls()
        self._params = gm.ckpts.load_params(params_path)
        self._tokenizer = self._build_tokenizer(gm, params_path)
        self._sampler = gm.text.ChatSampler(
            model=self._model,
            params=self._params,
            tokenizer=self._tokenizer,
            multi_turn=self._gen.multi_turn,
        )

        self.config = ModelConfig(model_name=str(params_path))
        self._cache = SampleCache(self.config)

    @classmethod
    def from_kagglehub(
        cls,
        kagglehub_slug: str,
        model_class_name: str = "Gemma4_E2B",
        **kwargs,
    ) -> "Gemma4FlaxModel":
        """Download Flax params from KaggleHub and instantiate a Gemma4FlaxModel."""
        import kagglehub  # type: ignore

        path = kagglehub.model_download(kagglehub_slug)
        return cls(params_path=path, model_class_name=model_class_name, **kwargs)

    @staticmethod
    def _resolve_model_class(gm, name: str):
        try:
            return getattr(gm.nn, name)
        except AttributeError as err:
            available = ", ".join(n for n in dir(gm.nn) if n.startswith("Gemma4"))
            raise ValueError(
                f"gm.nn has no class {name!r}. Available Gemma 4 classes: {available}"
            ) from err

    @staticmethod
    def _build_tokenizer(gm, path):
        for factory in (
            lambda: gm.text.Tokenizer.from_pretrained(path),
            lambda: gm.text.Tokenizer(path),
        ):
            try:
                return factory()
            except (AttributeError, TypeError):
                continue
        raise RuntimeError(
            "Could not instantiate gm.text.Tokenizer — the `gemma` library's "
            "constructor API may have changed."
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_length(self) -> int:
        return 131072

    @property
    def add_special_tokens(self) -> bool:
        return False

    def greedy_until(self, docs, **kwargs) -> List[ModelResponse]:
        responses = []
        for doc in docs:
            if not self._gen.multi_turn and hasattr(self._sampler, "reset"):
                self._sampler.reset()
            text = self._sampler.chat(
                doc.query,
                max_tokens=self._gen.max_new_tokens,
                temperature=self._gen.temperature,
                top_p=self._gen.top_p,
                top_k=self._gen.top_k,
            )
            responses.append(ModelResponse(text=[text]))
        return responses

    def loglikelihood(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemma4FlaxModel does not implement loglikelihood — "
            "pick a task that uses greedy_until."
        )

    def loglikelihood_rolling(self, *args, **kwargs):
        raise NotImplementedError("Gemma4FlaxModel does not implement loglikelihood_rolling.")

    def __repr__(self) -> str:
        return (
            f"Gemma4FlaxModel(model_path={self.model_path!r}, "
            f"model_class={self.model_class_name!r})"
        )
