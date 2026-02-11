# HKD 汇率抓取、预测与历史回测一体化框架

本项目已重构为统一流水线框架，支持你随时运行并完成：

1. 增量抓取最新招商银行汇率（默认港币）
2. 执行 LSTM 多特征直接多步预测（默认未来 30 天）
3. 回测所有历史预测批次（预测 vs 实际）
4. 保存历史对比图与报告（按批次归档）
5. 同步最新结果到 `分析/` 目录（兼容旧使用习惯）

---

## 一次运行全链路

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium
python -m hkd_fx.cli run-all
```

推荐命令会自动执行：
- `抓取 -> 主仓更新 -> 预测 -> 注册 -> 历史回测 -> latest 镜像 -> 分析目录镜像`

---

## 工作区文件框架（完整）

```text
HKD_exchange_rate_crawling_and_analysis/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ commands.txt
├─ .gitignore
├─ config/
│  └─ default.yaml
├─ src/
│  └─ hkd_fx/
│     ├─ cli.py                       # 命令行入口（run-all / ingest / forecast / backtest）
│     ├─ app.py                       # 全链路编排与 latest/分析 镜像
│     ├─ contracts.py                 # 核心数据契约
│     ├─ config.py                    # 配置加载
│     ├─ utils.py
│     ├─ ingestion/
│     │  ├─ cmb_client.py             # Playwright 抓取客户端
│     │  ├─ parser.py                 # 表格解析
│     │  └─ incremental_sync.py       # 增量抓取并入主仓
│     ├─ dataset/
│     │  ├─ canonical_store.py        # 主数据仓读写与去重
│     │  ├─ registry.py               # 预测/回测注册表
│     │  └─ snapshot_store.py         # 抓取快照保存
│     ├─ forecast/
│     │  ├─ features.py               # 特征工程与缩放
│     │  ├─ lstm_model.py             # LSTM 训练/评估/预测
│     │  ├─ render_forecast.py        # 预测图表渲染
│     │  └─ pipeline.py               # 预测流水线
│     ├─ backtest/
│     │  ├─ evaluator.py              # 指标计算与对齐
│     │  ├─ compare_renderer.py       # 对比图与报告渲染
│     │  └─ orchestrator.py           # 历史批次回测调度
│     └─ reporting/
│        └─ forecast_report.py        # 预测报告生成
├─ scripts/
│  ├─ run_all.py                      # 便捷脚本入口
│  ├─ migrate_legacy_data.py          # 迁移旧数据到主仓
│  └─ rebuild_backtests.py            # 重建全部回测结果
├─ tests/
│  ├─ test_canonical_store.py
│  └─ test_evaluator.py
├─ 分析/
│  ├─ fit_forecast_lstm.py            # 旧入口兼容包装
│  ├─ compare_forecast_actual.py      # 旧入口兼容包装
│  ├─ trend_plot.py
│  └─ 未来30天预测.* / 预测对比*.*      # 最新镜像输出
├─ scrape_cmb_rates_playwright.py     # 旧入口兼容包装
├─ data/                              # 运行时生成（已在 .gitignore）
│  ├─ master/hkd_rates_master.csv
│  ├─ registry/forecast_runs.csv
│  ├─ registry/comparison_status.csv
│  └─ snapshots/crawl/*.csv
└─ artifacts/                         # 运行时生成（已在 .gitignore）
   ├─ forecasts/<forecast_run_id>/
   ├─ comparisons/<forecast_run_id>/
   └─ latest/
```

---

## 新目录结构（核心）

```text
config/default.yaml                # 全局配置
src/hkd_fx/                        # 新框架代码
data/master/hkd_rates_master.csv   # 规范化主数据仓
data/registry/forecast_runs.csv    # 预测批次注册表
data/registry/comparison_status.csv# 回测状态注册表
data/snapshots/crawl/*.csv         # 每次抓取快照
artifacts/forecasts/<run_id>/      # 每次预测产物归档
artifacts/comparisons/<run_id>/    # 每次回测产物归档
artifacts/latest/                  # 最新结果镜像
分析/                               # 中文兼容输出镜像
```

---

## 常用命令

### 1) 全链路（推荐）

```bash
python -m hkd_fx.cli run-all
```

可选参数：

```bash
python -m hkd_fx.cli run-all --currency 港币 --rate-col 汇卖价 --pages 5 --lookback 30 --horizon 30
```

### 2) 仅抓取

```bash
python -m hkd_fx.cli ingest --currency 港币 --pages 5
```

### 3) 仅预测

```bash
python -m hkd_fx.cli forecast --rate-col 汇卖价 --lookback 30 --horizon 30
```

### 4) 回测全部历史预测

```bash
python -m hkd_fx.cli backtest
```

---

## 输出说明

### 预测归档（每次预测一个目录）

- `artifacts/forecasts/<forecast_run_id>/forecast.csv`
- `artifacts/forecasts/<forecast_run_id>/forecast.png`
- `artifacts/forecasts/<forecast_run_id>/forecast.html`
- `artifacts/forecasts/<forecast_run_id>/forecast_report.txt`

### 回测归档（每个预测批次一个目录）

- `artifacts/comparisons/<forecast_run_id>/comparison.csv`
- `artifacts/comparisons/<forecast_run_id>/comparison.png`
- `artifacts/comparisons/<forecast_run_id>/comparison.html`
- `artifacts/comparisons/<forecast_run_id>/comparison_report.txt`

### 最新镜像

- `artifacts/latest/*`
- `分析/未来30天预测.*`
- `分析/预测对比*.*`

---

## 历史数据迁移

重构后首次可运行：

```bash
python scripts/migrate_legacy_data.py
```

该脚本会把 `hkd_rates_30pages.csv` 与 `actual_rates_recent.csv` 合并入主数据仓。

---

## 兼容旧脚本

以下旧入口仍可执行，内部转发到新框架：

- `python scrape_cmb_rates_playwright.py ...`
- `python 分析/fit_forecast_lstm.py ...`
- `python 分析/compare_forecast_actual.py`

---

## 配置文件

默认配置在 `config/default.yaml`，可配置：
- 币种、目标列、抓取页数
- lookback/horizon
- 主仓与归档路径

如需自定义配置：

```bash
python -m hkd_fx.cli run-all --config config/default.yaml
```
