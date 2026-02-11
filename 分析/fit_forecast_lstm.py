import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hkd_fx.app import run_forecast_only


def main() -> None:
    parser = argparse.ArgumentParser(description="兼容入口：LSTM 多特征直接多步预测")
    parser.add_argument("--input", default=None, help="兼容参数，新框架统一从主数据仓读取")
    parser.add_argument("--outdir", default=None, help="兼容参数，新框架输出到 artifacts/ 并镜像到 分析/")
    parser.add_argument("--rate_col", default=None, help="目标列，默认 汇卖价")
    parser.add_argument("--current_date", default=None, help="兼容参数，新框架默认使用主仓最新日期")
    parser.add_argument("--lookback", type=int, default=None, help="回看窗口天数")
    args = parser.parse_args()

    result = run_forecast_only(rate_col=args.rate_col, lookback=args.lookback)
    print(f"预测CSV: {result.forecast_csv}")
    print(f"预测PNG: {result.forecast_png}")
    print(f"预测HTML: {result.forecast_html}")
    print(f"预测报告: {result.report_txt}")
    print("提示: 建议改用 `hkd-fx run-all`，可自动完成抓取、预测、回测与镜像。")


if __name__ == "__main__":
    main()
