from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    diff = forecast - actual
    abs_diff = np.abs(diff)

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(abs_diff))
    mape = float(np.mean(abs_diff / (actual + 1e-8)) * 100)

    if len(actual) <= 1:
        direction_acc = None
    else:
        actual_dir = np.sign(np.diff(actual))
        forecast_dir = np.sign(np.diff(forecast))
        direction_acc = float(np.mean(actual_dir == forecast_dir) * 100)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "方向准确率(%)": direction_acc,
        "最大误差": float(np.max(abs_diff)),
        "最小误差": float(np.min(abs_diff)),
        "平均误差": float(np.mean(diff)),
    }


def build_comparison_frame(actual_df: pd.DataFrame, forecast_df: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    actual = actual_df[["date", rate_col]].copy().rename(columns={"date": "日期", rate_col: "实际值"})
    forecast = forecast_df[["日期", "预测值"]].copy()
    forecast["日期"] = pd.to_datetime(forecast["日期"], errors="coerce")

    actual["日期"] = pd.to_datetime(actual["日期"], errors="coerce")
    merged = pd.merge(actual, forecast, on="日期", how="inner")
    merged = merged.dropna(subset=["日期", "实际值", "预测值"]).sort_values("日期").reset_index(drop=True)
    return merged

