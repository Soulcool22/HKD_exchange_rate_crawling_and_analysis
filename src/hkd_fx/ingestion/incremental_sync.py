from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..contracts import PipelineConfig
from ..dataset.canonical_store import (
    load_master_rates,
    merge_master,
    normalize_crawl_rows,
    save_master_rates,
)
from ..dataset.snapshot_store import write_crawl_snapshot
from .cmb_client import scrape_pages_sync


@dataclass(frozen=True)
class IngestionResult:
    snapshot_file: Path
    master_file: Path
    crawled_rows: int
    incoming_rows: int
    merged_rows: int
    added_rows: int


def run_incremental_sync(
    *,
    config: PipelineConfig,
    root: Path,
    pipeline_run_id: str,
    pages_override: int | None = None,
    currency_override: str | None = None,
) -> IngestionResult:
    currency = currency_override or config.currency
    pages = pages_override if pages_override is not None else config.crawl_pages

    rows = scrape_pages_sync(base_url=config.source_url, currency=currency, max_pages=pages)

    snapshot_file = root / config.data_dir / config.crawl_snapshot_dir / f"{pipeline_run_id}.csv"
    write_crawl_snapshot(snapshot_file, rows)

    snapshot_df = pd.read_csv(snapshot_file, encoding="utf-8-sig")
    incoming = normalize_crawl_rows(snapshot_df, currency=currency, source_run_id=pipeline_run_id)
    master_before = load_master_rates(config, root)
    merged = merge_master(master_before, incoming)
    master_file = save_master_rates(config, root, merged)

    added_rows = max(0, len(merged) - len(master_before))
    return IngestionResult(
        snapshot_file=snapshot_file,
        master_file=master_file,
        crawled_rows=len(rows),
        incoming_rows=len(incoming),
        merged_rows=len(merged),
        added_rows=added_rows,
    )

