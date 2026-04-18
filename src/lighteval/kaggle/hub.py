# Push a Gemma4EvalResult to a HuggingFace Hub dataset repo.
#
# This is an optional publish step — Kaggle users can evaluate and inspect
# results locally without ever touching this module. When they do want to
# publish, one call uploads the run directory (parquets + summary.json +
# report.md + visual_report.html) to an HF dataset repo.
from __future__ import annotations

import json
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
    """Upload a Gemma4EvalResult's output directory to the HuggingFace Hub.

    Creates the dataset repo if it doesn't exist, then uploads the entire
    run directory (details parquets, summary.json, report.md, and the
    visual_report.html if `result.save_report()` was called).

    Parameters
    ----------
    result : Gemma4EvalResult
        The outcome object from `Gemma4Eval().run()`.
    repo_id : str
        Target dataset repo, e.g. 'my-user/gemma4-eval-results'.
    private : bool, default False
        Create the repo as private. Ignored if the repo already exists.
    commit_message : str, optional
        Overrides the auto-generated commit message.
    token : str, optional
        HF Hub access token. Falls back to HF_TOKEN env var / `huggingface-cli login`.

    Returns
    -------
    str : the URL of the uploaded folder.
    """

    from huggingface_hub import HfApi, create_repo

    if result.output_dir is None:
        raise RuntimeError(
            "result.output_dir is unset — did you call result.save_report() "
            "or run the pipeline to completion?"
        )

    output_dir = Path(result.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"result.output_dir does not exist: {output_dir}")

    # Ensure a summary.json + report.md exist in the output dir; call
    # save_report() if they don't so the uploaded folder is self-describing.
    needs_save = not any(
        (output_dir / name).exists()
        for name in ("summary.json", "report.md")
    )
    if needs_save and hasattr(result, "save_report"):
        result.save_report()

    api = HfApi(token=token)

    # Create the repo (no-op if it exists).
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
