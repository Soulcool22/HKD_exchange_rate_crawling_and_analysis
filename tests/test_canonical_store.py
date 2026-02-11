from __future__ import annotations

import pandas as pd

from hkd_fx.dataset.canonical_store import merge_master, normalize_crawl_rows


def test_normalize_crawl_rows_and_merge_deduplicate() -> None:
    incoming_raw = pd.DataFrame(
        [
            {"日期": "2025年12月01日", "汇买价": "90.1", "钞买价": "90.1", "汇卖价": "90.5", "钞卖价": "90.5"},
            {"日期": "2025年12月02日", "汇买价": "90.2", "钞买价": "90.2", "汇卖价": "90.6", "钞卖价": "90.6"},
        ]
    )
    incoming = normalize_crawl_rows(incoming_raw, currency="港币", source_run_id="run_new")

    existing = pd.DataFrame(
        [
            {
                "date": "2025-12-01",
                "currency": "港币",
                "汇买价": 90.0,
                "钞买价": 90.0,
                "汇卖价": 90.4,
                "钞卖价": 90.4,
                "fetched_at": "2025-12-01T00:00:00",
                "source_run_id": "run_old",
            }
        ]
    )

    merged = merge_master(existing, incoming)
    assert len(merged) == 2

    day1 = merged[merged["date"] == pd.Timestamp("2025-12-01")].iloc[0]
    assert day1["source_run_id"] == "run_new"
    assert abs(float(day1["汇卖价"]) - 90.5) < 1e-8

