# Model-source resolution for Kaggle users.
#
# Kaggle notebooks typically refer to models by one of three handles:
#   1. A KaggleHub slug                  'google/gemma-4/transformers/gemma-4-e2b-it'
#   2. A local /kaggle/input directory   '/kaggle/input/models/.../1'
#   3. A HuggingFace Hub repo id         'lthn/lemer-hf-bf16'
#
# This module collapses those into a single `resolve_model_source` helper
# so the Kaggle API surface accepts any of them without the user having
# to know which is which.
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _is_local_path(source: str) -> bool:
    """Crude check: exists on disk AND looks like a directory path."""
    return os.path.sep in source and Path(source).exists()


def _is_kagglehub_slug(source: str) -> bool:
    """KaggleHub model slugs have at least three slash-separated parts:
    owner/model/framework (e.g. 'google/gemma-4/transformers/gemma-4-e2b-it').
    HF Hub repo ids typically have one or two parts ('org/repo')."""
    parts = source.strip("/").split("/")
    return len(parts) >= 3


def resolve_model_source(source: str, label: Optional[str] = None) -> str:
    """Resolve a model source string to a local filesystem path.

    - Local paths are returned as-is.
    - KaggleHub slugs are pulled via kagglehub.model_download.
    - HF Hub repo ids are returned as-is (transformers handles the download).

    Parameters
    ----------
    source : str
        Local path, KaggleHub slug, or HF Hub repo id.
    label : str, optional
        Printed in progress messages so parallel resolves are readable
        ('base', 'test', etc.).
    """
    tag = f"[{label}] " if label else ""

    if _is_local_path(source):
        print(f"{tag}local path: {source}")
        return source

    if _is_kagglehub_slug(source):
        try:
            import kagglehub  # type: ignore
        except ImportError as err:
            raise ImportError(
                "kagglehub is required to resolve KaggleHub model slugs. "
                "Install with `pip install kagglehub`."
            ) from err

        print(f"{tag}pulling from KaggleHub: {source}")
        resolved = kagglehub.model_download(source)
        print(f"{tag}resolved: {resolved}")
        return resolved

    # Treat as HF Hub repo id — transformers handles caching/download.
    print(f"{tag}HuggingFace Hub repo: {source}")
    return source
