import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _read_master_series(rate_col: str):
    try:
        from hkd_fx.config import load_config, resolve_path
        from hkd_fx.dataset.canonical_store import load_master_rates
    except Exception:
        return None

    config = load_config()
    root = resolve_path(".")
    master = load_master_rates(config, root)
    if master.empty or rate_col not in master.columns:
        return None
    master = master[master["currency"] == config.currency].copy()
    master["date"] = pd.to_datetime(master["date"], errors="coerce")
    master = master.dropna(subset=["date"])
    master[rate_col] = pd.to_numeric(master[rate_col], errors="coerce")
    master = master.dropna(subset=[rate_col]).sort_values("date")
    return master.set_index("date")[rate_col]


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def read_csv_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp936", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err


def load_series(input_path: Path, start_date: str, end_date: str, rate_col: str = "汇买价") -> pd.Series:
    df = read_csv_fallback(input_path)
    # 解析中文日期格式，例如：2025年11月3日
    df["日期"] = pd.to_datetime(df["日期"], format="%Y年%m月%d日", errors="coerce")
    # 数值列转为浮点
    for col in ["汇买价", "钞买价", "汇卖价", "钞卖价"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 过滤时间范围
    mask = (df["日期"] >= pd.to_datetime(start_date)) & (df["日期"] <= pd.to_datetime(end_date))
    df = df.loc[mask].copy()
    df.sort_values("日期", inplace=True)
    series = df.set_index("日期")[rate_col].dropna()
    # 按天补齐缺失并前向填充，保证等间隔
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index).ffill()
    series.name = rate_col
    return series


def plot_trend(series: pd.Series, outdir: Path, title: str) -> Path:
    plt.figure(figsize=(12, 4))
    # 兼容中文字体（若系统无可用中文字体会自动回退）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(series.index, series.values, color="#1f77b4", linewidth=1.2)
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel(series.name)
    plt.grid(alpha=0.3)
    out_path = outdir / "趋势图.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"趋势图已生成：{out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="港币汇率趋势图生成（仅趋势图）")
    parser.add_argument("--input", default=str(Path("hkd_rates_30pages.csv")), help="输入CSV路径")
    parser.add_argument("--outdir", default=str(Path("分析")), help="输出目录")
    parser.add_argument("--start", default="2022-09-21", help="起始日期，YYYY-MM-DD")
    parser.add_argument("--end", default="2025-11-03", help="结束日期，YYYY-MM-DD")
    parser.add_argument("--rate_col", default="汇买价", help="列名：如 汇买价/汇卖价 等")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    master_series = _read_master_series(args.rate_col)
    if master_series is not None:
        mask = (master_series.index >= pd.to_datetime(args.start)) & (master_series.index <= pd.to_datetime(args.end))
        series = master_series.loc[mask]
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
        series = series.reindex(full_index).ffill()
    else:
        series = load_series(input_path, args.start, args.end, rate_col=args.rate_col)

    title = f"{args.start} 至 {args.end} 趋势（{args.rate_col}）"
    plot_trend(series, outdir, title=title)


if __name__ == "__main__":
    main()
