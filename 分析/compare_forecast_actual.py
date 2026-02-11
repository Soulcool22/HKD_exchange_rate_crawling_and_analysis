import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hkd_fx.app import run_backtest_only


def main() -> None:
    parser = argparse.ArgumentParser(description="兼容入口：预测值与实际值对比分析")
    parser.add_argument("--actual", default=None, help="兼容参数，新框架从主数据仓读取实际值")
    parser.add_argument("--forecast", default=None, help="兼容参数，新框架按注册表遍历预测批次")
    parser.add_argument("--rate_col", default=None, help="兼容参数，默认使用配置")
    parser.add_argument("--outdir", default=None, help="兼容参数，新框架输出到 artifacts/comparisons")
    _ = parser.parse_args()

    results = run_backtest_only()
    print(f"回测批次数: {len(results)}")
    for item in results:
        print(f"- {item.forecast_run_id}: status={item.status}, overlap_days={item.overlap_days}, dir={item.comparison_dir}")


if __name__ == "__main__":
    main()
