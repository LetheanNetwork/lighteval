from lighteval.models.transformers import Gemma4Model, GenerationConfig

from .loader import resolve_model_source
from .pipeline import Gemma4Eval, Gemma4EvalResult
from .tracker import KaggleEvaluationTracker


def __getattr__(name):
    if name == "Gemma4FlaxModel":
        from lighteval.models.flax import Gemma4FlaxModel

        return Gemma4FlaxModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Gemma4Eval",
    "Gemma4EvalResult",
    "Gemma4Model",
    "Gemma4FlaxModel",
    "GenerationConfig",
    "KaggleEvaluationTracker",
    "resolve_model_source",
]
