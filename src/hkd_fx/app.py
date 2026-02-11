from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import load_config, resolve_path
from .contracts import PipelineConfig
from .utils import make_run_id

if TYPE_CHECKING:
    from .backtest.orchestrator import BacktestResult
    from .forecast.pipeline import ForecastResult
    from .ingestion.incremental_sync import IngestionResult


@dataclass(frozen=True)
class RunAllResult:
    pipeline_run_id: str
    forecast_run_id: str
    ingestion: IngestionResult
    forecast: ForecastResult
    backtests: list[BacktestResult]


def _project_root() -> Path:
    return resolve_path(".")


def _artifact_forecast_dir(config: PipelineConfig, root: Path, forecast_run_id: str) -> Path:
    return root / config.artifacts_dir / "forecasts" / forecast_run_id


def _mirror_latest(config: PipelineConfig, root: Path, forecast: ForecastResult, backtests: list[BacktestResult]) -> None:
    latest_dir = root / config.artifacts_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(forecast.forecast_csv, latest_dir / "forecast.csv")
    shutil.copy2(forecast.forecast_png, latest_dir / "forecast.png")
    shutil.copy2(forecast.forecast_html, latest_dir / "forecast.html")
    shutil.copy2(forecast.report_txt, latest_dir / "forecast_report.txt")

    completed = [item for item in backtests if item.status in {"completed", "partial"}]
    if completed:
        latest_cmp = sorted(completed, key=lambda item: item.forecast_run_id)[-1]
        cmp_dir = latest_cmp.comparison_dir
        for source, target in [
            (cmp_dir / "comparison.csv", latest_dir / "comparison.csv"),
            (cmp_dir / "comparison.png", latest_dir / "comparison.png"),
            (cmp_dir / "comparison.html", latest_dir / "comparison.html"),
            (cmp_dir / "comparison_report.txt", latest_dir / "comparison_report.txt"),
        ]:
            if source.exists():
                shutil.copy2(source, target)

    analysis_dir = root / config.analysis_mirror_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mirror_pairs = [
        (latest_dir / "forecast.csv", analysis_dir / "未来30天预测.csv"),
        (latest_dir / "forecast.png", analysis_dir / "未来30天预测.png"),
        (latest_dir / "forecast.html", analysis_dir / "未来30天预测.html"),
        (latest_dir / "forecast_report.txt", analysis_dir / "预测报告.txt"),
        (latest_dir / "comparison.csv", analysis_dir / "预测对比数据.csv"),
        (latest_dir / "comparison.png", analysis_dir / "预测对比图.png"),
        (latest_dir / "comparison.html", analysis_dir / "预测对比图.html"),
        (latest_dir / "comparison_report.txt", analysis_dir / "预测对比报告.txt"),
    ]
    for source, target in mirror_pairs:
        if source.exists():
            shutil.copy2(source, target)


def run_all(
    *,
    config_path: str | None = None,
    currency: str | None = None,
    rate_col: str | None = None,
    pages: int | None = None,
    lookback: int | None = None,
    horizon: int | None = None,
) -> RunAllResult:
    from .backtest.orchestrator import run_backtest_for_all
    from .dataset.registry import append_forecast_registry, build_forecast_meta
    from .forecast.pipeline import run_forecast_pipeline
    from .ingestion.incremental_sync import run_incremental_sync

    config = load_config(config_path)
    root = _project_root()

    pipeline_run_id = make_run_id("pipeline")
    forecast_run_id = make_run_id("forecast")

    ingestion = run_incremental_sync(
        config=config,
        root=root,
        pipeline_run_id=pipeline_run_id,
        pages_override=pages,
        currency_override=currency,
    )

    forecast_dir = _artifact_forecast_dir(config, root, forecast_run_id)
    forecast = run_forecast_pipeline(
        config=config,
        root=root,
        pipeline_run_id=pipeline_run_id,
        forecast_run_id=forecast_run_id,
        output_dir=forecast_dir,
        rate_col_override=rate_col,
        lookback_override=lookback,
        horizon_override=horizon,
    )

    meta = build_forecast_meta(
        forecast_run_id=forecast_run_id,
        pipeline_run_id=pipeline_run_id,
        currency=currency or config.currency,
        rate_col=rate_col or config.rate_col,
        model_name=config.model_name,
        lookback=lookback or config.lookback,
        horizon=horizon or config.horizon,
        train_start=forecast.train_start.date(),
        train_end=forecast.train_end.date(),
        forecast_start=forecast.forecast_series.index.min().date(),
        forecast_end=forecast.forecast_series.index.max().date(),
        forecast_dir=str(forecast.output_dir),
        source_snapshot_file=str(ingestion.snapshot_file),
        rmse_scaled=float(forecast.metrics["rmse"]),
        mae_scaled=float(forecast.metrics["mae"]),
        mape_scaled=float(forecast.metrics["mape"]),
    )
    append_forecast_registry(config, root, meta)

    backtests = run_backtest_for_all(config, root)
    _mirror_latest(config, root, forecast, backtests)

    return RunAllResult(
        pipeline_run_id=pipeline_run_id,
        forecast_run_id=forecast_run_id,
        ingestion=ingestion,
        forecast=forecast,
        backtests=backtests,
    )


def run_ingest_only(*, config_path: str | None = None, currency: str | None = None, pages: int | None = None) -> IngestionResult:
    from .ingestion.incremental_sync import run_incremental_sync

    config = load_config(config_path)
    root = _project_root()
    pipeline_run_id = make_run_id("pipeline")
    return run_incremental_sync(
        config=config,
        root=root,
        pipeline_run_id=pipeline_run_id,
        pages_override=pages,
        currency_override=currency,
    )


def run_forecast_only(
    *,
    config_path: str | None = None,
    rate_col: str | None = None,
    lookback: int | None = None,
    horizon: int | None = None,
) -> ForecastResult:
    from .backtest.orchestrator import run_backtest_for_all
    from .dataset.registry import append_forecast_registry, build_forecast_meta
    from .forecast.pipeline import run_forecast_pipeline

    config = load_config(config_path)
    root = _project_root()
    pipeline_run_id = make_run_id("pipeline")
    forecast_run_id = make_run_id("forecast")
    forecast_dir = _artifact_forecast_dir(config, root, forecast_run_id)

    forecast = run_forecast_pipeline(
        config=config,
        root=root,
        pipeline_run_id=pipeline_run_id,
        forecast_run_id=forecast_run_id,
        output_dir=forecast_dir,
        rate_col_override=rate_col,
        lookback_override=lookback,
        horizon_override=horizon,
    )

    meta = build_forecast_meta(
        forecast_run_id=forecast_run_id,
        pipeline_run_id=pipeline_run_id,
        currency=config.currency,
        rate_col=rate_col or config.rate_col,
        model_name=config.model_name,
        lookback=lookback or config.lookback,
        horizon=horizon or config.horizon,
        train_start=forecast.train_start.date(),
        train_end=forecast.train_end.date(),
        forecast_start=forecast.forecast_series.index.min().date(),
        forecast_end=forecast.forecast_series.index.max().date(),
        forecast_dir=str(forecast.output_dir),
        source_snapshot_file="",
        rmse_scaled=float(forecast.metrics["rmse"]),
        mae_scaled=float(forecast.metrics["mae"]),
        mape_scaled=float(forecast.metrics["mape"]),
    )
    append_forecast_registry(config, root, meta)

    backtests = run_backtest_for_all(config, root)
    _mirror_latest(config, root, forecast, backtests)
    return forecast


def run_backtest_only(*, config_path: str | None = None) -> list[BacktestResult]:
    from .backtest.orchestrator import run_backtest_for_all

    config = load_config(config_path)
    root = _project_root()
    return run_backtest_for_all(config, root)
