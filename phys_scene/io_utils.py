from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .schemas import to_builtin


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_builtin(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def discover_images(root: Path, pattern: str) -> list[Path]:
    root = root.expanduser().resolve()
    matches = list(root.glob(pattern))
    if not matches:
        stem = pattern.replace(".HEIC", "").replace(".heic", "")
        matches = []
        for suffix in ("*.HEIC", "*.heic", "*.png", "*.jpg", "*.jpeg"):
            matches.extend(root.glob(stem + suffix.replace("*", "")))
    image_exts = {".heic", ".heif", ".png", ".jpg", ".jpeg"}
    deduped = {p.resolve() for p in matches if p.suffix.lower() in image_exts}
    return sorted(deduped, key=natural_key)


def compact_list(values: Iterable[int]) -> list[int]:
    return sorted({int(v) for v in values})
