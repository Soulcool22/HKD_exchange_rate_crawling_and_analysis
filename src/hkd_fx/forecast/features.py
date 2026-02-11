from __future__ import annotations

import numpy as np
import pandas as pd


def as_daily_continuous(series: pd.Series) -> pd.Series:
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_index).ffill()


def build_features(series: pd.Series) -> pd.DataFrame:
    idx = series.index
    day_of_week = pd.Series(idx.dayofweek, index=idx)
    day_of_month = pd.Series(idx.day, index=idx)

    df = pd.DataFrame(
        {
            "y": series.astype(np.float32).values,
            "dow_sin": np.sin(2 * np.pi * day_of_week / 7.0).astype(np.float32).values,
            "dow_cos": np.cos(2 * np.pi * day_of_week / 7.0).astype(np.float32).values,
            "dom_sin": np.sin(2 * np.pi * day_of_month / 31.0).astype(np.float32).values,
            "dom_cos": np.cos(2 * np.pi * day_of_month / 31.0).astype(np.float32).values,
            "delta1": series.diff().fillna(0.0).astype(np.float32).values,
            "ma7_diff": (series - series.rolling(7, min_periods=1).mean()).astype(np.float32).values,
            "std7": series.rolling(7, min_periods=1).std().fillna(0.0).astype(np.float32).values,
            "is_weekend": (day_of_week >= 5).astype(np.float32).values,
        },
        index=idx,
    )
    return df


def scale_features(df: pd.DataFrame) -> np.ndarray:
    result = df.copy()
    for col in ["delta1", "ma7_diff", "std7"]:
        if col in result.columns:
            mean = float(np.nanmean(result[col].values))
            std = float(np.nanstd(result[col].values))
            std = std if std > 1e-8 else 1.0
            result[col] = ((result[col] - mean) / std).astype(np.float32)
    return result.astype(np.float32).values


def scale_series(series: pd.Series):
    values = series.values.astype(np.float32)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max - value_min < 1e-12:
        scaled = np.zeros_like(values)
    else:
        scaled = (values - value_min) / (value_max - value_min)
    return scaled, value_min, value_max


def inverse_scale(arr, value_min, value_max):
    return arr * (value_max - value_min) + value_min


def make_supervised_multistep(features: np.ndarray, target_scaled: np.ndarray, lookback: int, horizon: int):
    x_list, y_list = [], []
    total = len(target_scaled)
    n_features = features.shape[1]
    for i in range(total - lookback - horizon + 1):
        x_list.append(features[i : i + lookback, :])
        y_list.append(target_scaled[i + lookback : i + lookback + horizon])
    x_data = np.array(x_list, dtype=np.float32).reshape((-1, lookback, n_features))
    y_data = np.array(y_list, dtype=np.float32).reshape((-1, horizon))
    return x_data, y_data

