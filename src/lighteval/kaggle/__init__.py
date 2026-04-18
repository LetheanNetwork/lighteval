import importlib.util
import subprocess
import sys
from typing import TYPE_CHECKING


def _ensure_notebook_deps() -> None:
    try:
        in_notebook = __import__("IPython").get_ipython() is not None
    except (ImportError, AttributeError):
        in_notebook = False
    if not in_notebook:
        return
    for pkg in ("nbformat",):
        if importlib.util.find_spec(pkg) is None:
            print(f"[lighteval.kaggle] installing missing notebook dep: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_ensure_notebook_deps()


from lighteval.models.transformers import Gemma4Model, GenerationConfig  # noqa: E402

from .loader import resolve_model_source  # noqa: E402
from .pipeline import Gemma4Eval, Gemma4EvalResult  # noqa: E402
from .tracker import KaggleEvaluationTracker  # noqa: E402


if TYPE_CHECKING:
    from lighteval.models.flax import Gemma4FlaxModel


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
