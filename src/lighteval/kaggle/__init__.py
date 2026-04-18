"""Kaggle-first API for lighteval.

This package wraps lighteval's core Pipeline + EvaluationTracker in a
shape that matches how Kaggle users actually work: KaggleHub-resolved
model paths, `/kaggle/working` output defaults, dual-GPU paired runs out
of the box, and an optional HF Hub publish step.

Typical notebook usage (5 cells, end-to-end):

    # Cell 1 — settings
    RUN_NAME = 'my-gemma4-run'
    BASE = 'google/gemma-4/transformers/gemma-4-e2b-it'
    TEST = 'my-user/my-gemma4-finetune'

    # Cell 2 — install
    # !pip install -q 'lighteval @ git+https://github.com/LetheanNetwork/lighteval.git@gemma4'

    # Cell 3 — run
    from lighteval.kaggle import Gemma4Eval
    run = Gemma4Eval(base=BASE, test=TEST, task='mmlu_pro', rounds=8).run()

    # Cell 4 — visualise
    run.dashboard()

    # Cell 5 — publish to HF Hub (optional)
    run.push_to_hub('my-user/gemma4-eval-results')
"""

from .loader import resolve_model_source
from .tracker import KaggleEvaluationTracker
from .pipeline import Gemma4Eval, Gemma4EvalResult

__all__ = [
    "Gemma4Eval",
    "Gemma4EvalResult",
    "KaggleEvaluationTracker",
    "resolve_model_source",
]
