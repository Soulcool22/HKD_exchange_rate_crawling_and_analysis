from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..contracts import ComparisonRunMeta, PipelineConfig
from ..dataset.canonical_store import load_master_rates
from ..dataset.registry import (
    forecast_registry_to_list,
    load_forecast_registry,
    upsert_comparison_registry,
)
from .compare_renderer import (
    write_comparison_csv,
    write_comparison_html,
    write_comparison_png,
    write_comparison_report,
)
from .evaluator import build_comparison_frame, compute_metrics


@dataclass(frozen=True)
class BacktestResult:
    forecast_run_id: str
    status: str
    overlap_days: int
    comparison_dir: Path


def _load_forecast_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _actual_currency_frame(config: PipelineConfig, root: Path) -> pd.DataFrame:
    master = load_master_rates(config, root)
    if master.empty:
        return pd.DataFrame(columns=["date", config.rate_col])
    frame = master[master["currency"] == config.currency].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[config.rate_col] = pd.to_numeric(frame[config.rate_col], errors="coerce")
    frame = frame.dropna(subset=["date", config.rate_col]).sort_values("date")
    return frame[["date", config.rate_col]]


def run_backtest_for_all(config: PipelineConfig, root: Path) -> list[BacktestResult]:
    registry = load_forecast_registry(config, root)
    runs = forecast_registry_to_list(registry)
    actual = _actual_currency_frame(config, root)

    results: list[BacktestResult] = []
    artifacts_root = root / config.artifacts_dir / "comparisons"

    for run in runs:
        run_id = run["forecast_run_id"]
        forecast_dir = Path(run["forecast_dir"])
        forecast_csv = forecast_dir / "forecast.csv"
        comparison_dir = artifacts_root / run_id
        comparison_dir.mkdir(parents=True, exist_ok=True)

        try:
            forecast_df = _load_forecast_csv(forecast_csv)
            merged = build_comparison_frame(actual, forecast_df, config.rate_col)
            if merged.empty:
                meta = ComparisonRunMeta(
                    forecast_run_id=run_id,
                    updated_at=datetime.now(),
                    status="pending_actuals",
                    overlap_days=0,
                    actual_start=None,
                    actual_end=None,
                    rmse=None,
                    mae=None,
                    mape=None,
                    direction_acc=None,
                    comparison_dir=str(comparison_dir),
                    notes="暂无可用重叠实际数据",
                )
                upsert_comparison_registry(config, root, meta)
                results.append(BacktestResult(run_id, "pending_actuals", 0, comparison_dir))
                continue

            metrics = compute_metrics(merged["实际值"].values, merged["预测值"].values)
            write_comparison_png(merged, metrics, comparison_dir / "comparison.png")
            write_comparison_html(merged, metrics, comparison_dir / "comparison.html")
            write_comparison_report(merged, metrics, comparison_dir / "comparison_report.txt")
            write_comparison_csv(merged, comparison_dir / "comparison.csv")

            overlap = len(merged)
            forecast_days = len(forecast_df)
            status = "completed" if overlap >= forecast_days else "partial"

            meta = ComparisonRunMeta(
                forecast_run_id=run_id,
                updated_at=datetime.now(),
                status=status,
                overlap_days=overlap,
                actual_start=merged["日期"].min().date(),
                actual_end=merged["日期"].max().date(),
                rmse=metrics["RMSE"],
                mae=metrics["MAE"],
                mape=metrics["MAPE(%)"],
                direction_acc=metrics["方向准确率(%)"],
                comparison_dir=str(comparison_dir),
                notes="",
            )
            upsert_comparison_registry(config, root, meta)
            results.append(BacktestResult(run_id, status, overlap, comparison_dir))
        except Exception as error:
            meta = ComparisonRunMeta(
                forecast_run_id=run_id,
                updated_at=datetime.now(),
                status="failed",
                overlap_days=0,
                actual_start=None,
                actual_end=None,
                rmse=None,
                mae=None,
                mape=None,
                direction_acc=None,
                comparison_dir=str(comparison_dir),
                notes=str(error),
            )
            upsert_comparison_registry(config, root, meta)
            results.append(BacktestResult(run_id, "failed", 0, comparison_dir))

    return results
