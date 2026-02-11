from __future__ import annotations

from datetime import datetime
from pathlib import Path


def now_utc() -> datetime:
    return datetime.utcnow()


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

