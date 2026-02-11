from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..contracts import ComparisonRunMeta, ForecastRunMeta, PipelineConfig


FORECAST_REGISTRY_COLUMNS = [
    "forecast_run_id",
    "created_at",
    "pipeline_run_id",
    "currency",
    "rate_col",
    "model_name",
    "lookback",
    "horizon",
    "train_start",
    "train_end",
    "forecast_start",
    "forecast_end",
    "forecast_dir",
    "source_snapshot_file",
    "rmse_scaled",
    "mae_scaled",
    "mape_scaled",
]

COMPARISON_REGISTRY_COLUMNS = [
    "forecast_run_id",
    "updated_at",
    "status",
    "overlap_days",
    "actual_start",
    "actual_end",
    "rmse",
    "mae",
    "mape",
    "direction_acc",
    "comparison_dir",
    "notes",
]


def _load_registry(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, encoding="utf-8-sig")
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df[columns]


def _save_registry(path: Path, df: pd.DataFrame, columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()[columns]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_forecast_registry(config: PipelineConfig, root: Path) -> pd.DataFrame:
    return _load_registry(root / config.data_dir / config.forecast_registry_file, FORECAST_REGISTRY_COLUMNS)


def append_forecast_registry(config: PipelineConfig, root: Path, meta: ForecastRunMeta) -> Path:
    path = root / config.data_dir / config.forecast_registry_file
    registry = _load_registry(path, FORECAST_REGISTRY_COLUMNS)
    row = pd.DataFrame(
        [
            {
                "forecast_run_id": meta.forecast_run_id,
                "created_at": meta.created_at.isoformat(timespec="seconds"),
                "pipeline_run_id": meta.pipeline_run_id,
                "currency": meta.currency,
                "rate_col": meta.rate_col,
                "model_name": meta.model_name,
                "lookback": meta.lookback,
                "horizon": meta.horizon,
                "train_start": meta.train_start.isoformat(),
                "train_end": meta.train_end.isoformat(),
                "forecast_start": meta.forecast_start.isoformat(),
                "forecast_end": meta.forecast_end.isoformat(),
                "forecast_dir": meta.forecast_dir,
                "source_snapshot_file": meta.source_snapshot_file,
                "rmse_scaled": meta.rmse_scaled,
                "mae_scaled": meta.mae_scaled,
                "mape_scaled": meta.mape_scaled,
            }
        ]
    )
    if registry.empty:
        registry = row.copy()
    else:
        registry = pd.concat([registry, row], ignore_index=True)
    return _save_registry(path, registry, FORECAST_REGISTRY_COLUMNS)


def load_comparison_registry(config: PipelineConfig, root: Path) -> pd.DataFrame:
    return _load_registry(root / config.data_dir / config.comparison_registry_file, COMPARISON_REGISTRY_COLUMNS)


def upsert_comparison_registry(config: PipelineConfig, root: Path, meta: ComparisonRunMeta) -> Path:
    path = root / config.data_dir / config.comparison_registry_file
    registry = _load_registry(path, COMPARISON_REGISTRY_COLUMNS)

    row = {
        "forecast_run_id": meta.forecast_run_id,
        "updated_at": meta.updated_at.isoformat(timespec="seconds"),
        "status": meta.status,
        "overlap_days": meta.overlap_days,
        "actual_start": meta.actual_start.isoformat() if meta.actual_start else None,
        "actual_end": meta.actual_end.isoformat() if meta.actual_end else None,
        "rmse": meta.rmse,
        "mae": meta.mae,
        "mape": meta.mape,
        "direction_acc": meta.direction_acc,
        "comparison_dir": meta.comparison_dir,
        "notes": meta.notes,
    }

    exists = registry[registry["forecast_run_id"] == meta.forecast_run_id]
    if exists.empty:
        if registry.empty:
            registry = pd.DataFrame([row], columns=COMPARISON_REGISTRY_COLUMNS)
        else:
            registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    else:
        idx = exists.index[0]
        for key, value in row.items():
            registry.at[idx, key] = value

    return _save_registry(path, registry, COMPARISON_REGISTRY_COLUMNS)


def forecast_registry_to_list(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    sorted_df = df.copy()
    sorted_df["created_at"] = pd.to_datetime(sorted_df["created_at"], errors="coerce")
    sorted_df = sorted_df.sort_values("created_at")
    return sorted_df.to_dict(orient="records")


def build_forecast_meta(
    *,
    forecast_run_id: str,
    pipeline_run_id: str,
    currency: str,
    rate_col: str,
    model_name: str,
    lookback: int,
    horizon: int,
    train_start,
    train_end,
    forecast_start,
    forecast_end,
    forecast_dir: str,
    source_snapshot_file: str,
    rmse_scaled: float,
    mae_scaled: float,
    mape_scaled: float,
) -> ForecastRunMeta:
    return ForecastRunMeta(
        forecast_run_id=forecast_run_id,
        created_at=datetime.now(),
        pipeline_run_id=pipeline_run_id,
        currency=currency,
        rate_col=rate_col,
        model_name=model_name,
        lookback=lookback,
        horizon=horizon,
        train_start=train_start,
        train_end=train_end,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        forecast_dir=forecast_dir,
        source_snapshot_file=source_snapshot_file,
        rmse_scaled=rmse_scaled,
        mae_scaled=mae_scaled,
        mape_scaled=mape_scaled,
    )
