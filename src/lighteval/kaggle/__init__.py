import importlib.util
import logging
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


def _ensure_transformers_gemma4() -> None:
    # Kaggle's base image can satisfy `transformers>=4.54` without actually
    # shipping the Gemma 4 architecture (the gemma4 model_type landed in a
    # later minor). Detect the gap by inspecting CONFIG_MAPPING_NAMES and
    # force an upgrade if needed. Module eviction lets the next import pick
    # up the new version when transformers hasn't been imported yet.
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except ImportError:
        return
    if "gemma4" in CONFIG_MAPPING_NAMES:
        return

    print("[lighteval.kaggle] transformers lacks gemma4 support — upgrading")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"]
    )

    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            del sys.modules[name]


def _suppress_expected_warnings() -> None:
    # Gemma4Eval always sets max_samples — it's a paired A/B slice by design,
    # not a benchmark submission — so the upstream "THESE NUMBERS ARE ONLY
    # PARTIAL" warning is expected noise. The dashboard makes the slice scope
    # explicit for anyone reading the output.
    pipeline_logger = logging.getLogger("lighteval.pipeline")

    class _MaxSamplesFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "max_samples WAS SET" not in record.getMessage()

    if not any(isinstance(f, _MaxSamplesFilter) for f in pipeline_logger.filters):
        pipeline_logger.addFilter(_MaxSamplesFilter())


_ensure_notebook_deps()
_ensure_transformers_gemma4()
_suppress_expected_warnings()


from lighteval.models.transformers import Gemma4Model, GenerationConfig  # noqa: E402

from .loader import resolve_model_source  # noqa: E402
from .pipeline import Gemma4Eval, Gemma4EvalResult  # noqa: E402
from .tracker import KaggleEvaluationTracker  # noqa: E402


if TYPE_CHECKING:
    from lighteval.models.flax import Gemma4FlaxModel
    from lighteval.models.mlx import Gemma4MLXModel


def __getattr__(name):
    if name == "Gemma4FlaxModel":
        from lighteval.models.flax import Gemma4FlaxModel

        return Gemma4FlaxModel
    if name == "Gemma4MLXModel":
        from lighteval.models.mlx import Gemma4MLXModel

        return Gemma4MLXModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Gemma4Eval",
    "Gemma4EvalResult",
    "Gemma4Model",
    "Gemma4FlaxModel",
    "Gemma4MLXModel",
    "GenerationConfig",
    "KaggleEvaluationTracker",
    "resolve_model_source",
]
