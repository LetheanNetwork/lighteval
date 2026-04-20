from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lighteval.models.transformers.gemma4_model import (
    Gemma4Model,
    GenerationConfig,
)
from lighteval.pipeline import (
    ParallelismManager,
    Pipeline,
    PipelineParameters,
)

from .loader import resolve_model_source
from .tracker import KaggleEvaluationTracker


_TASK_ALIASES = {
    "mmlu_pro": "mmlu_pro|0",
    "global_mmlu": "global_mmlu_full_en|0",
    "ifeval": "ifeval|0",
}


def _expand_task(task: str) -> str:
    if "|" in task:
        return task
    return _TASK_ALIASES.get(task, task)


@dataclass
class Gemma4EvalResult:
    run_name: str
    task: str
    base_source: str
    test_source: str
    rounds: int
    n_questions: int = 1
    samples_start: int = 0
    hardware_plan: Optional[str] = None
    base_detail_paths: List[str] = field(default_factory=list)
    test_detail_paths: List[str] = field(default_factory=list)
    output_dir: Optional[Path] = None

    _analysis: Optional[tuple] = field(default=None, repr=False)

    def analyze(self):
        """Parse the paired details parquets into (detail_df, question_summaries, totals)."""
        if self._analysis is not None:
            return self._analysis
        from .analyze import analyze_pair

        self._analysis = analyze_pair(
            base_paths=self.base_detail_paths,
            test_paths=self.test_detail_paths,
            base_model_name=self.base_source,
            test_model_name=self.test_source,
            run_name=self.run_name,
            task=self.task,
            samples_start=self.samples_start,
        )
        return self._analysis

    def dashboard(self, **kwargs) -> str:
        """Render the comparison dashboard inline and return the HTML."""
        from .dashboard import render as _render

        return _render(self, **kwargs)

    def dump_research(self, *, include_responses: bool = True) -> Path:
        """Write per-question prompt+response markdown under `<output_dir>/research`."""
        from .research import dump as _dump

        return _dump(self, include_responses=include_responses)

    def save_report(self) -> Path:
        """Write summary.csv, summary.json, report.md, and visual_report.html to the output directory."""
        import datetime as dt

        import pandas as pd

        if self.output_dir is None:
            raise RuntimeError(
                "output_dir is unset — run() must complete before save_report()."
            )

        detail_df, question_summaries, totals = self.analyze()
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        detail_df.to_parquet(out / "comparison_details.parquet", index=False)

        pd.DataFrame(
            [
                {"side": side, **values}
                for side, values in totals.items()
                if isinstance(values, dict)
            ]
        ).to_csv(out / "summary.csv", index=False)

        (out / "summary.json").write_text(
            json.dumps(
                {
                    "run_name": self.run_name,
                    "task": self.task,
                    "base_source": self.base_source,
                    "test_source": self.test_source,
                    "rounds": self.rounds,
                    "n_questions": self.n_questions,
                    "samples_start": self.samples_start,
                    "totals": totals,
                    "questions": question_summaries,
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
                indent=2,
            )
        )

        lines: List[str] = [
            f"# Gemma 4 Eval: {self.run_name}",
            "",
            f"- task: `{self.task}`",
            f"- base: `{self.base_source}`",
            f"- test: `{self.test_source}`",
            f"- window: samples_start `{self.samples_start}`, "
            f"questions `{self.n_questions}`, rounds `{self.rounds}`",
            "",
            "| Side | Model | Samples | Correct | Per-round acc | Majority acc |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for side in ("base", "test"):
            t = totals[side]
            lines.append(
                f"| {side} | `{t['model']}` | {t['samples']} | {t['correct']} | "
                f"{t['per_round_accuracy_pct']:.2f}% | {t['majority_accuracy_pct']:.2f}% |"
            )
        lines.append("")
        lines.append(f"Per-round delta: **{totals['delta_pp']:+.2f} pp**")
        lines.append(f"Majority delta: **{totals['majority_delta_pp']:+.2f} pp**")
        lines.append("")
        for q in question_summaries:
            lines.append(f"## Q{q['question_index']}: {q['question'][:140]}")
            lines.append(f"Gold: **{q['gold_letter']}** {q['gold_text']}")
            lines.append(
                f"- base answers: `{q['base']['answers']}` hits "
                f"{sum(q['base']['hits'])}/{q['base']['total']}"
            )
            lines.append(
                f"- test answers: `{q['test']['answers']}` hits "
                f"{sum(q['test']['hits'])}/{q['test']['total']}"
            )
            lines.append("")
        (out / "report.md").write_text("\n".join(lines))

        plotly_sections = getattr(self, "_dashboard_plotly_html", [])
        if plotly_sections:
            (out / "plotly_report.html").write_text(
                "<!doctype html><html><head><meta charset=\"utf-8\">"
                "<title>Gemma 4 Eval Plots</title></head><body>"
                + "".join(plotly_sections)
                + "</body></html>"
            )

        print(f"Saved report under {out}")
        return out

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        commit_message: Optional[str] = None,
        token: Optional[str] = None,
    ) -> str:
        """Upload the run directory to a HuggingFace Hub dataset repo."""
        from .hub import push_result as _push

        return _push(
            self,
            repo_id,
            private=private,
            commit_message=commit_message,
            token=token,
        )


class Gemma4Eval:
    """Paired evaluation of two Gemma 4 models over the same task and sampling recipe."""

    def __init__(
        self,
        base,
        test,
        task: str = "mmlu_pro",
        rounds: int = 1,
        n_questions: int = 1,
        samples_start: int = 0,
        generation: Optional[GenerationConfig] = None,
        device_map: str = "auto",
        dtype: str = "auto",
        parallel: bool | str = False,  # Default to False! Sharing GPUs is chaotic.
        run_name: Optional[str] = None,
        backend: str = "auto",
        research: bool = False,
    ):
        self._base_arg = base
        self._test_arg = test
        self.base_source = self._source_label(base)
        self.test_source = self._source_label(test)
        self.task = _expand_task(task)
        self.rounds = rounds
        self.n_questions = n_questions
        self.samples_start = samples_start
        self.generation = generation or GenerationConfig()
        self.device_map = device_map
        self.dtype = dtype
        self.parallel = parallel
        self.run_name = run_name or self._default_run_name(self.base_source, self.test_source)
        self.backend = self._resolve_backend(backend)
        self.research = research

    @staticmethod
    def _source_label(source) -> str:
        if isinstance(source, str):
            return source
        if hasattr(source, "model_path"):
            return str(source.model_path)
        if callable(source):
            return getattr(source, "__name__", "factory")
        return type(source).__name__

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend not in ("auto", "transformers", "mlx"):
            raise ValueError(f"backend must be one of 'auto', 'transformers', 'mlx' (got {backend!r})")
        if backend != "auto":
            return backend
        try:
            from lighteval.models.mlx.gemma4_mlx_model import is_apple_silicon
        except ImportError:
            return "transformers"
        if is_apple_silicon():
            try:
                import mlx_lm  # noqa: F401
            except ImportError:
                return "transformers"
            return "mlx"
        return "transformers"

    @staticmethod
    def _default_run_name(base, test) -> str:
        def tail(source) -> str:
            if isinstance(source, str):
                return source.rstrip("/").split("/")[-1]
            return getattr(source, "model_path", type(source).__name__).rstrip("/").split("/")[-1]

        return f"gemma4-{tail(base)}-vs-{tail(test)}"

    @staticmethod
    def _resolve_side(source, label: str):
        """String → model path (via loader); instance or factory → pass through."""
        if isinstance(source, str):
            return resolve_model_source(source, label=label)
        return source

    def run(self) -> Gemma4EvalResult:
        """Resolve models, run all rounds, and return a populated Gemma4EvalResult."""
        print(f"=== Gemma4Eval: {self.run_name} ===")
        base_path = self._resolve_side(self._base_arg, label="base")
        test_path = self._resolve_side(self._test_arg, label="test")

        num_gpus = self._visible_gpu_count()
        devices = self._list_devices(num_gpus)
        
        # Determine if we should run in parallel
        use_parallel = False
        if use_parallel == "auto":
            if num_gpus >= 2 and isinstance(self._base_arg, str) and isinstance(self._test_arg, str):
                use_parallel = True
            else:
                use_parallel = False
        
        # Safety net: If parallel=True was explicitly forced, verify we actually have the hardware.
        if use_parallel and num_gpus < 2:
            print("Warning: parallel=True requested, but fewer than 2 GPUs found. Falling back to sequential execution.")
            use_parallel = False

        print(f"Devices: {devices}  parallel={use_parallel}")

        result = Gemma4EvalResult(
            run_name=self.run_name,
            task=self.task,
            base_source=self.base_source,
            test_source=self.test_source,
            rounds=self.rounds,
            n_questions=self.n_questions,
            samples_start=self.samples_start,
            hardware_plan=(
                f"parallel: base on GPU 0, test on GPU 1 ({self.backend})"
                if use_parallel
                else f"sequential: backend={self.backend}, device_map={self.device_map}"
            ),
        )

        for round_idx in range(1, self.rounds + 1):
            if use_parallel:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fb = ex.submit(self._run_one, base_path, "base", round_idx, gpu_index=0)
                    ft = ex.submit(self._run_one, test_path, "test", round_idx, gpu_index=1)
                    result.base_detail_paths.append(fb.result())
                    result.test_detail_paths.append(ft.result())
            else:
                result.base_detail_paths.append(
                    self._run_one(base_path, "base", round_idx, gpu_index=None)
                )
                result.test_detail_paths.append(
                    self._run_one(test_path, "test", round_idx, gpu_index=None)
                )

        # Locate the tracker's output dir via the side from the last round.
        if result.base_detail_paths:
            sample = Path(result.base_detail_paths[-1])
            # details parquet lives at <out>/details/<task>/<run>.parquet — climb up
            out_dir = sample
            while out_dir.parent != out_dir and out_dir.name != "details":
                out_dir = out_dir.parent
            result.output_dir = out_dir.parent.parent

        if self.research:
            result.dump_research()

        print(f"=== run complete — details under {result.output_dir} ===")
        return result

    def _run_one(
        self,
        model_or_path,
        side: str,
        round_idx: int,
        gpu_index: Optional[int],
    ) -> str:
        import inspect

        from lighteval.models.abstract_model import LightevalModel
        from lighteval.models.transformers.gemma4_model import _pick_device

        device_map = f"cuda:{gpu_index}" if gpu_index is not None else self.device_map
        concrete_device = _pick_device(device_map)
        round_name = f"{self.run_name}/{side}_round{round_idx}"
        tracker = KaggleEvaluationTracker(run_name=round_name)
        out_dir = tracker.output_dir_path

        if out_dir.exists() and any(out_dir.iterdir()):
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        params = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            max_samples=self.n_questions,
            samples_start=self.samples_start,
        )

        if isinstance(model_or_path, str):
            model = self._build_model(model_or_path, device_map)
            source = "path"
            we_own = True
        elif isinstance(model_or_path, LightevalModel):
            model = model_or_path
            source = "preloaded"
            we_own = False
        elif callable(model_or_path):
            sig = inspect.signature(model_or_path)
            accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            if "device" in sig.parameters or accepts_kwargs:
                model = model_or_path(device=concrete_device)
            else:
                model = model_or_path()
            source = "factory"
            we_own = True
        else:
            raise TypeError(
                f"[{side}] expected a string, a LightevalModel instance, or a callable "
                f"returning one — got {type(model_or_path).__name__}"
            )

        print(
            f"[{side}] round {round_idx}/{self.rounds}  backend={self.backend}  "
            f"device={concrete_device}  source={source}"
        )
        pipeline = Pipeline(
            tasks=self.task,
            pipeline_parameters=params,
            evaluation_tracker=tracker,
            model=model,
        )
        pipeline.evaluate()
        pipeline.save_and_push_results()

        if we_own:
            pipeline.model = None  # break the back-ref so gc can release weights
            del model, pipeline
            self._reclaim_gpu_memory()

        parquets = sorted(out_dir.glob("details/**/*.parquet"))
        if not parquets:
            raise RuntimeError(
                f"[{side}] no details parquet produced in {out_dir} — "
                f"check the lighteval output above for errors."
            )
        return str(parquets[0])

    def _build_model(self, model_path: str, device_map: str):
        if self.backend == "mlx":
            from lighteval.models.mlx import Gemma4MLXModel

            return Gemma4MLXModel(model_path=model_path, generation=self.generation)
        return Gemma4Model(
            model_path=model_path,
            device_map=device_map,
            dtype=self.dtype,
            generation=self.generation,
        )

    @staticmethod
    def _visible_gpu_count() -> int:
        try:
            import torch  # type: ignore

            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            return 0

    @staticmethod
    def _list_devices(num_gpus: int) -> List[str]:
        if num_gpus > 0:
            return [f"cuda:{i}" for i in range(num_gpus)]
        try:
            import torch  # type: ignore

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return ["mps"]
        except ImportError:
            pass
        return ["cpu"]

    @staticmethod
    def _reclaim_gpu_memory() -> None:
        import gc

        gc.collect()
        try:
            import torch  # type: ignore
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            current = torch.mps.current_allocated_memory() / 1e9
            driver = torch.mps.driver_allocated_memory() / 1e9
            print(
                f"[mps] tensors={current:.2f} GB · allocator reserved={driver:.2f} GB "
                f"· gap={driver - current:.2f} GB (allocator cache, recoverable)"
            )
