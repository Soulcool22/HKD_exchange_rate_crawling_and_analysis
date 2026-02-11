from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hkd_fx.app import run_backtest_only


def main() -> None:
    results = run_backtest_only()
    print(f"rebuild_backtests: total={len(results)}")
    for item in results:
        print(f"{item.forecast_run_id} -> {item.status} ({item.overlap_days} days)")


if __name__ == "__main__":
    main()
