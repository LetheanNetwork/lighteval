# Copyright 2025 LetheanNetwork
# Licensed under the MIT License (see LICENSE)
#
# Gemma 4 support via Google DeepMind's official `gemma` library (JAX/Flax).
#
# The reference Gemma 4 implementation lives at
# https://github.com/google-deepmind/gemma. Install from source while the
# package is being polished:
#
#     pip install git+https://github.com/google-deepmind/gemma.git
#
# Usage from a notebook:
#
#     from lighteval.models.flax import Gemma4FlaxModel, GenerationConfig
#     import kagglehub
#     params = kagglehub.model_download('google/gemma-4/flax/gemma-4-e2b-it')
#     model = Gemma4FlaxModel(params, model_class_name='Gemma4_E2B')
#     # pass into Pipeline(..., model=model)
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.models.transformers.gemma4_model import GenerationConfig


class Gemma4FlaxModel(LightevalModel):
    """LightevalModel wrapping the reference Flax Gemma 4 implementation.

    Uses `gm.text.ChatSampler` under the hood so the model sees Gemma 4's
    exact chat template with the same defaults Google uses in their
    reference examples. Set `generation.multi_turn = True` if you want
    the sampler to preserve conversation history across docs (you usually
    do not for evaluation).

    Parameters
    ----------
    params_path : str
        Local directory holding the flax params — e.g. the output of
        `kagglehub.model_download('google/gemma-4/flax/gemma-4-e2b-it')`.
    model_class_name : str, default 'Gemma4_E2B'
        Name of the `gm.nn` architecture class. Use 'Gemma4_E4B' for E4B
        and `dir(gm.nn)` to list other Gemma 4 classes available on your
        `gemma` install.
    generation : GenerationConfig | None
        Sampling config. Defaults to Google's Gemma 4 calibrated recipe.
    """

    def __init__(
        self,
        params_path: str,
        model_class_name: str = "Gemma4_E2B",
        generation: Optional[GenerationConfig] = None,
    ):
        from gemma import gm  # lazy import so the transformers path works without it

        self.model_path = params_path
        self.model_class_name = model_class_name
        self._gen = generation or GenerationConfig()

        cls = self._resolve_model_class(gm, model_class_name)
        self._model = cls()
        # gm.ckpts.load_params accepts a local checkpoint directory on recent
        # main branches. Earlier versions take a CheckpointPath enum — if
        # you pin an older version, swap this for the enum form.
        self._params = gm.ckpts.load_params(params_path)
        self._tokenizer = self._build_tokenizer(gm, params_path)
        self._sampler = gm.text.ChatSampler(
            model=self._model,
            params=self._params,
            tokenizer=self._tokenizer,
            multi_turn=self._gen.multi_turn,
        )

    # ------------------------------------------------------------------
    # Construction helpers.
    # ------------------------------------------------------------------
    @classmethod
    def from_kagglehub(
        cls,
        kagglehub_slug: str,
        model_class_name: str = "Gemma4_E2B",
        **kwargs,
    ) -> "Gemma4FlaxModel":
        """Construct a Gemma4FlaxModel from a KaggleHub flax slug.

        Example
        -------
        >>> model = Gemma4FlaxModel.from_kagglehub(
        ...     'google/gemma-4/flax/gemma-4-e2b-it',
        ...     model_class_name='Gemma4_E2B',
        ... )
        """
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
        """Instantiate a gm.text.Tokenizer, tolerating minor API shape drift."""
        for factory in (
            lambda: gm.text.Tokenizer.from_pretrained(path),
            lambda: gm.text.Tokenizer(path),
        ):
            try:
                return factory()
            except (AttributeError, TypeError):
                continue
        raise RuntimeError(
            "Could not instantiate gm.text.Tokenizer — the `gemma` lib's API "
            "may have moved. Check the main branch for the current constructor."
        )

    # --- Abstract properties ---
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_length(self) -> int:
        return 131072

    @property
    def add_special_tokens(self) -> bool:
        return False

    # --- Inference ---
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
