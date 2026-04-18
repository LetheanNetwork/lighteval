# Kaggle-sane EvaluationTracker — defaults that match how notebook users run.
#
# The stock EvaluationTracker requires several decisions (push to hub? push
# to tensorboard? which results path template?) that Kaggle users usually
# don't need up-front. This wrapper fixes sane defaults:
#   - output_dir under /kaggle/working when available
#   - save_details=True (the Kaggle run should always produce inspectable parquets)
#   - push_to_hub=False by default (user opts in via Gemma4EvalResult.push_to_hub)
#   - no tensorboard/wandb/nanotron cruft
from __future__ import annotations

from pathlib import Path
from typing import Optional

from lighteval.logging.evaluation_tracker import EvaluationTracker


def default_output_root() -> Path:
    """Where results should live by default.

    On Kaggle this is `/kaggle/working`. Elsewhere we default to a
    `./runs` directory next to the notebook.
    """
    kaggle_root = Path("/kaggle/working")
    if kaggle_root.exists():
        return kaggle_root
    return Path("runs")


class KaggleEvaluationTracker(EvaluationTracker):
    """EvaluationTracker with Kaggle-friendly defaults.

    Parameters
    ----------
    run_name : str
        Used to form the output directory ({root}/{run_name}).
    output_root : Path | None
        Override the root. Defaults to `/kaggle/working` or `./runs`.
    save_details : bool, default True
        Persist per-question parquet details — required for the dashboard.
    """

    def __init__(
        self,
        run_name: str,
        output_root: Optional[Path] = None,
        save_details: bool = True,
    ):
        root = Path(output_root) if output_root is not None else default_output_root()
        out_dir = root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            output_dir=str(out_dir),
            save_details=save_details,
            push_to_hub=False,
            push_to_tensorboard=False,
            use_wandb=False,
        )
        self.run_name = run_name
        self.output_dir_path = out_dir
