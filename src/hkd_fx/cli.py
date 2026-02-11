from __future__ import annotations

import argparse

from .app import run_all, run_backtest_only, run_forecast_only, run_ingest_only


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HKD 汇率抓取-预测-回测统一入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="配置文件路径，默认 config/default.yaml")

    run_all_parser = subparsers.add_parser("run-all", parents=[common], help="执行抓取+预测+回测全链路")
    run_all_parser.add_argument("--currency", default=None, help="币种，默认取配置")
    run_all_parser.add_argument("--rate-col", default=None, help="目标列，默认取配置")
    run_all_parser.add_argument("--pages", type=int, default=None, help="抓取页数，默认取配置")
    run_all_parser.add_argument("--lookback", type=int, default=None, help="回看窗口天数")
    run_all_parser.add_argument("--horizon", type=int, default=None, help="预测天数")

    ingest_parser = subparsers.add_parser("ingest", parents=[common], help="仅执行增量抓取")
    ingest_parser.add_argument("--currency", default=None, help="币种，默认取配置")
    ingest_parser.add_argument("--pages", type=int, default=None, help="抓取页数")

    forecast_parser = subparsers.add_parser("forecast", parents=[common], help="仅执行预测")
    forecast_parser.add_argument("--rate-col", default=None, help="目标列")
    forecast_parser.add_argument("--lookback", type=int, default=None, help="回看窗口")
    forecast_parser.add_argument("--horizon", type=int, default=None, help="预测天数")

    subparsers.add_parser("backtest", parents=[common], help="回测全部历史预测批次")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-all":
        result = run_all(
            config_path=args.config,
            currency=args.currency,
            rate_col=args.rate_col,
            pages=args.pages,
            lookback=args.lookback,
            horizon=args.horizon,
        )
        print(f"pipeline_run_id={result.pipeline_run_id}")
        print(f"forecast_run_id={result.forecast_run_id}")
        print(f"crawled_rows={result.ingestion.crawled_rows} incoming_rows={result.ingestion.incoming_rows} added_rows={result.ingestion.added_rows}")
        print(f"forecast_csv={result.forecast.forecast_csv}")
        print(f"forecast_png={result.forecast.forecast_png}")
        print(f"forecast_html={result.forecast.forecast_html}")
        print(f"report_txt={result.forecast.report_txt}")
        print(f"backtest_runs={len(result.backtests)}")
        return

    if args.command == "ingest":
        result = run_ingest_only(config_path=args.config, currency=args.currency, pages=args.pages)
        print(f"snapshot_file={result.snapshot_file}")
        print(f"master_file={result.master_file}")
        print(f"crawled_rows={result.crawled_rows} incoming_rows={result.incoming_rows} added_rows={result.added_rows}")
        return

    if args.command == "forecast":
        result = run_forecast_only(
            config_path=args.config,
            rate_col=args.rate_col,
            lookback=args.lookback,
            horizon=args.horizon,
        )
        print(f"forecast_csv={result.forecast_csv}")
        print(f"forecast_png={result.forecast_png}")
        print(f"forecast_html={result.forecast_html}")
        print(f"report_txt={result.report_txt}")
        return

    if args.command == "backtest":
        results = run_backtest_only(config_path=args.config)
        print(f"backtest_runs={len(results)}")
        for item in results:
            print(f"{item.forecast_run_id}: status={item.status}, overlap_days={item.overlap_days}, dir={item.comparison_dir}")
        return


if __name__ == "__main__":
    main()

