from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..contracts import PipelineConfig


MASTER_COLUMNS = [
    "date",
    "currency",
    "汇买价",
    "钞买价",
    "汇卖价",
    "钞卖价",
    "fetched_at",
    "source_run_id",
]


def _read_csv_fallback(path: Path, encodings: tuple[str, ...]) -> pd.DataFrame:
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as error:  # pragma: no cover
            last_error = error
            continue
    raise last_error


def _parse_date(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    normalized = (
        series.str.replace("年", "-", regex=False)
        .str.replace("月", "-", regex=False)
        .str.replace("日", "", regex=False)
    )
    return pd.to_datetime(normalized, errors="coerce")


def load_master_rates(config: PipelineConfig, root: Path) -> pd.DataFrame:
    master_path = root / config.data_dir / config.master_rates_file
    if not master_path.exists():
        return pd.DataFrame(columns=MASTER_COLUMNS)

    df = _read_csv_fallback(master_path, config.encoding_fallbacks)
    missing = [column for column in MASTER_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"主数据仓缺少字段: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in ["汇买价", "钞买价", "汇卖价", "钞卖价"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_crawl_rows(rows_df: pd.DataFrame, currency: str, source_run_id: str) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    data = rows_df.copy()
    data = data.rename(columns={"日期": "date"})
    data["date"] = _parse_date(data["date"])
    data = data.dropna(subset=["date"])

    for col in ["汇买价", "钞买价", "汇卖价", "钞卖价"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["汇买价", "钞买价", "汇卖价", "钞卖价"])

    data["currency"] = currency
    data["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    data["source_run_id"] = source_run_id
    data = data[MASTER_COLUMNS]
    return data


def merge_master(master_df: pd.DataFrame, incoming_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        merged = incoming_df.copy()
    elif incoming_df.empty:
        merged = master_df.copy()
    else:
        merged = pd.concat([master_df, incoming_df], ignore_index=True)

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"])
    merged = merged.sort_values(["date", "fetched_at", "source_run_id"]).drop_duplicates(
        subset=["date", "currency"], keep="last"
    )
    merged = merged.sort_values(["date", "currency"]).reset_index(drop=True)
    return merged


def save_master_rates(config: PipelineConfig, root: Path, data: pd.DataFrame) -> Path:
    master_path = root / config.data_dir / config.master_rates_file
    master_path.parent.mkdir(parents=True, exist_ok=True)
    output = data.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output.to_csv(master_path, index=False, encoding="utf-8-sig")
    return master_path

