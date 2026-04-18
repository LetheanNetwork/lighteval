from __future__ import annotations

from pathlib import Path
from typing import Optional


def push_result(
    result,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Upload a Gemma4EvalResult's output directory to the HuggingFace Hub."""

    from huggingface_hub import HfApi, create_repo

    if result.output_dir is None:
        raise RuntimeError(
            "result.output_dir is unset — did you call result.save_report() "
            "or run the pipeline to completion?"
        )

    output_dir = Path(result.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"result.output_dir does not exist: {output_dir}")

    needs_save = not any(
        (output_dir / name).exists()
        for name in ("summary.json", "report.md")
    )
    if needs_save and hasattr(result, "save_report"):
        result.save_report()

    api = HfApi(token=token)

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    msg = commit_message or f"lighteval Gemma4 run: {result.run_name}"
    print(f"Uploading {output_dir} -> {repo_id} (dataset repo)")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=msg,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Published: {url}")
    return url
