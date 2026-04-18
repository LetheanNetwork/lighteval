from __future__ import annotations

from pathlib import Path
from typing import Optional

from lighteval.logging.evaluation_tracker import EvaluationTracker


def default_output_root() -> Path:
    """Return `/kaggle/working` when present, otherwise `./runs`."""
    kaggle_root = Path("/kaggle/working")
    if kaggle_root.exists():
        return kaggle_root
    return Path("runs")


class KaggleEvaluationTracker(EvaluationTracker):
    """EvaluationTracker pre-configured for notebook runs.

    Writes to `{output_root}/{run_name}` (defaulting to `/kaggle/working`
    or `./runs`) with details parquets enabled and hub/tensorboard/wandb
    side-channels disabled.
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
