from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ..contracts import PipelineConfig
from ..dataset.canonical_store import load_master_rates
from ..reporting.forecast_report import write_forecast_report
from .features import as_daily_continuous, build_features, inverse_scale, scale_features, scale_series
from .lstm_model import evaluate_recent_window, forecast_future
from .render_forecast import plot_forecast_png, write_forecast_html


@dataclass(frozen=True)
class ForecastResult:
    forecast_df: pd.DataFrame
    forecast_series: pd.Series
    history_tail: pd.Series
    metrics: dict
    output_dir: Path
    forecast_csv: Path
    forecast_png: Path
    forecast_html: Path
    report_txt: Path
    train_start: pd.Timestamp
    train_end: pd.Timestamp


def _build_trend_text(name: str, series: pd.Series) -> str:
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    net = float(end - start)
    pct = float((net / (start + 1e-8)) * 100.0)
    vmin = float(np.min(series.values))
    vmax = float(np.max(series.values))
    dmin = series.index[int(np.argmin(series.values))]
    dmax = series.index[int(np.argmax(series.values))]
    diffs = series.diff().dropna()
    mean_diff = float(np.mean(diffs.values)) if len(diffs) else 0.0
    std_diff = float(np.std(diffs.values)) if len(diffs) else 0.0
    pos = int(np.sum(diffs.values > 0))
    neg = int(np.sum(diffs.values < 0))

    return (
        f"{name}从 {series.index.min():%Y-%m-%d} 至 {series.index.max():%Y-%m-%d}，"
        f"起点 {start:.4f}，终点 {end:.4f}，净变动 {net:.4f}（{pct:.2f}%）。\n"
        f"最高值 {vmax:.4f} 出现在 {dmax:%Y-%m-%d}；最低值 {vmin:.4f} 出现在 {dmin:%Y-%m-%d}。\n"
        f"上涨天数 {pos}，下跌天数 {neg}，平均日变动 {mean_diff:.4f}，波动（日差标准差） {std_diff:.4f}。"
    )


def _load_rate_series(config: PipelineConfig, root: Path, rate_col_override: str | None = None) -> pd.Series:
    rate_col = rate_col_override or config.rate_col
    master = load_master_rates(config, root)
    if master.empty:
        raise ValueError("主数据仓为空，无法进行预测。")

    currency_data = master[master["currency"] == config.currency].copy()
    if currency_data.empty:
        raise ValueError(f"主数据仓中不存在币种 {config.currency} 的记录。")

    if rate_col not in currency_data.columns:
        raise ValueError(f"主数据仓中缺少目标列: {rate_col}")

    currency_data["date"] = pd.to_datetime(currency_data["date"], errors="coerce")
    currency_data = currency_data.dropna(subset=["date"])
    currency_data[rate_col] = pd.to_numeric(currency_data[rate_col], errors="coerce")
    currency_data = currency_data.dropna(subset=[rate_col]).sort_values("date")

    series = currency_data.set_index("date")[rate_col]
    series = series[~series.index.duplicated(keep="last")]
    return series


def run_forecast_pipeline(
    *,
    config: PipelineConfig,
    root: Path,
    pipeline_run_id: str,
    forecast_run_id: str,
    output_dir: Path,
    rate_col_override: str | None = None,
    lookback_override: int | None = None,
    horizon_override: int | None = None,
) -> ForecastResult:
    rate_col = rate_col_override or config.rate_col
    lookback = lookback_override if lookback_override is not None else config.lookback
    horizon = horizon_override if horizon_override is not None else config.horizon

    tf.random.set_seed(42)
    np.random.seed(42)

    raw = _load_rate_series(config, root, rate_col_override=rate_col)
    daily = as_daily_continuous(raw)

    features = build_features(daily)
    target = features["y"]
    x_scaled = scale_features(features.drop(columns=["y"]))
    target_scaled, value_min, value_max = scale_series(target)

    metrics = evaluate_recent_window(
        x_scaled,
        target_scaled,
        lookback=lookback,
        horizon=horizon,
        value_min=value_min,
        value_max=value_max,
    )
    forecast_scaled = forecast_future(x_scaled, target_scaled, lookback=lookback, steps=horizon)
    forecast_values = inverse_scale(forecast_scaled, value_min, value_max)

    train_end = pd.to_datetime(daily.index.max())
    forecast_index = pd.date_range(train_end + timedelta(days=1), periods=horizon, freq="D")
    forecast_series = pd.Series(forecast_values, index=forecast_index)
    forecast_df = pd.DataFrame({"日期": forecast_series.index.strftime("%Y-%m-%d"), "预测值": forecast_series.values})

    history_tail = daily.loc[train_end - timedelta(days=config.history_window_days - 1) : train_end]

    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_csv = output_dir / "forecast.csv"
    forecast_png = output_dir / "forecast.png"
    forecast_html = output_dir / "forecast.html"
    report_txt = output_dir / "forecast_report.txt"

    chart_title = (
        f"{train_end.strftime('%Y-%m-%d')} 至 {(train_end + timedelta(days=horizon)).strftime('%Y-%m-%d')} "
        f"趋势预测（{rate_col}，LSTM·多特征·直接多步）"
    )

    forecast_df.to_csv(forecast_csv, index=False, encoding="utf-8-sig")
    plot_forecast_png(history_tail, forecast_series, forecast_png, chart_title)
    write_forecast_html(history_tail, forecast_series, forecast_html, chart_title)

    report_info = {
        "pipeline_run_id": pipeline_run_id,
        "forecast_run_id": forecast_run_id,
        "current_date": train_end.strftime("%Y-%m-%d"),
        "rate_col": rate_col,
        "start_date": daily.index.min().strftime("%Y-%m-%d"),
        "end_date": train_end.strftime("%Y-%m-%d"),
        "lookback": lookback,
        "horizon": horizon,
        **metrics,
        "fc_start": forecast_index.min().strftime("%Y-%m-%d"),
        "fc_end": forecast_index.max().strftime("%Y-%m-%d"),
        "fc_min": float(np.min(forecast_series.values)),
        "fc_max": float(np.max(forecast_series.values)),
        "fc_change": float(forecast_series.values[-1] - forecast_series.values[0]),
        "actual_text": _build_trend_text("近15天实际", history_tail),
        "forecast_text": _build_trend_text("未来预测", forecast_series),
        "html_path": str(forecast_html),
        "png_path": str(forecast_png),
        "csv_path": str(forecast_csv),
        "report_path": str(report_txt),
    }
    write_forecast_report(report_txt, report_info)

    return ForecastResult(
        forecast_df=forecast_df,
        forecast_series=forecast_series,
        history_tail=history_tail,
        metrics=metrics,
        output_dir=output_dir,
        forecast_csv=forecast_csv,
        forecast_png=forecast_png,
        forecast_html=forecast_html,
        report_txt=report_txt,
        train_start=pd.to_datetime(daily.index.min()),
        train_end=train_end,
    )
