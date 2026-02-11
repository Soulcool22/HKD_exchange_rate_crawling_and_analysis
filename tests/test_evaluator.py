from __future__ import annotations

import pandas as pd

from hkd_fx.backtest.evaluator import build_comparison_frame, compute_metrics


def test_build_comparison_frame_and_metrics() -> None:
    actual = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-01", "2025-12-02", "2025-12-03"]),
            "汇卖价": [90.0, 90.5, 90.2],
        }
    )
    forecast = pd.DataFrame(
        {
            "日期": ["2025-12-02", "2025-12-03", "2025-12-04"],
            "预测值": [90.4, 90.3, 90.1],
        }
    )

    merged = build_comparison_frame(actual, forecast, "汇卖价")
    assert len(merged) == 2

    metrics = compute_metrics(merged["实际值"].values, merged["预测值"].values)
    assert metrics["RMSE"] >= 0
    assert metrics["MAE"] >= 0
    assert metrics["MAPE(%)"] >= 0

