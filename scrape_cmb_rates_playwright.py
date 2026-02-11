import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hkd_fx.app import run_ingest_only


def main() -> None:
    parser = argparse.ArgumentParser(description="兼容入口：增量抓取招商银行历史汇率")
    parser.add_argument("--nbr", default=None, help="货币中文名，例如：港币、美元、欧元等")
    parser.add_argument("--output", default=None, help="兼容参数，新框架下由数据仓管理，可忽略")
    parser.add_argument("--max_pages", type=int, default=None, help="最多抓取页数，默认使用配置")
    args = parser.parse_args()

    result = run_ingest_only(currency=args.nbr, pages=args.max_pages)
    print(f"抓取快照: {result.snapshot_file}")
    print(f"主数据仓: {result.master_file}")
    print(f"抓取条数: {result.crawled_rows}, 入库条数: {result.incoming_rows}, 新增日期: {result.added_rows}")
    if args.output:
        print("提示: --output 在新框架中不再直接输出单文件，已统一写入 data/master 与 snapshots。")


if __name__ == "__main__":
    main()
