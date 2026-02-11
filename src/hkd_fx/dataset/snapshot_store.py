from __future__ import annotations

from pathlib import Path

import pandas as pd


CRAWL_COLUMNS = ["日期", "汇买价", "钞买价", "汇卖价", "钞卖价"]


def write_crawl_snapshot(snapshot_file: Path, rows: list[dict]) -> Path:
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=CRAWL_COLUMNS)
    else:
        df = df.reindex(columns=CRAWL_COLUMNS)
    df.to_csv(snapshot_file, index=False, encoding="utf-8-sig")
    return snapshot_file

