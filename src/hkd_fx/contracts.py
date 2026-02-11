from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Literal


ComparisonStatus = Literal["pending_actuals", "partial", "completed", "failed"]


@dataclass(frozen=True)
class PipelineConfig:
    currency: str
    rate_col: str
    crawl_pages: int
    lookback: int
    horizon: int
    history_window_days: int
    model_name: str
    source_url: str
    data_dir: str
    artifacts_dir: str
    analysis_mirror_dir: str
    master_rates_file: str
    forecast_registry_file: str
    comparison_registry_file: str
    crawl_snapshot_dir: str
    encoding_fallbacks: tuple[str, ...]


@dataclass(frozen=True)
class RateRecord:
    date: date
    currency: str
    remittance_buy: float
    cash_buy: float
    remittance_sell: float
    cash_sell: float
    fetched_at: datetime
    source_run_id: str


@dataclass(frozen=True)
class ForecastPoint:
    forecast_run_id: str
    forecast_date: date
    forecast_value: float


@dataclass(frozen=True)
class ForecastRunMeta:
    forecast_run_id: str
    created_at: datetime
    pipeline_run_id: str
    currency: str
    rate_col: str
    model_name: str
    lookback: int
    horizon: int
    train_start: date
    train_end: date
    forecast_start: date
    forecast_end: date
    forecast_dir: str
    source_snapshot_file: str
    rmse_scaled: float
    mae_scaled: float
    mape_scaled: float


@dataclass(frozen=True)
class ComparisonRunMeta:
    forecast_run_id: str
    updated_at: datetime
    status: ComparisonStatus
    overlap_days: int
    actual_start: date | None
    actual_end: date | None
    rmse: float | None
    mae: float | None
    mape: float | None
    direction_acc: float | None
    comparison_dir: str
    notes: str

