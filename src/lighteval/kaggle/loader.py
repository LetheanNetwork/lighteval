from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _is_local_path(source: str) -> bool:
    return os.path.sep in source and Path(source).exists()


def _is_kagglehub_slug(source: str) -> bool:
    parts = source.strip("/").split("/")
    return len(parts) >= 3


def resolve_model_source(source: str, label: Optional[str] = None) -> str:
    """Resolve a local path, KaggleHub slug, or HF Hub repo id to a usable path.

    Local paths and HF repo ids are returned as-is; KaggleHub slugs are
    downloaded via `kagglehub.model_download`.
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

    print(f"{tag}HuggingFace Hub repo: {source}")
    return source
