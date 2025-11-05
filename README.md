# 招商银行历史汇率抓取与分析预测

此工作区为“数据爬取 + 分析预测”的复合结构：
- 数据爬取模块：
  - `scrape_cmb_rates_playwright.py`：Playwright 渲染并自动翻页抓取脚本
  - `requirements.txt`：依赖清单（抓取与分析通用）
  - `commands.txt`：常用安装与运行命令（含抓取与预测）
  - 示例数据：`hkd_rates_30pages.csv`（供分析预测脚本使用）
- 分析预测模块（`分析/`）：
  - `分析/fit_forecast_lstm.py`：多特征 + 直接多步 LSTM 预测
  - `分析/trend_plot.py`：仅生成历史趋势图
  - 产出文件：`分析/未来30天预测.html`、`分析/未来30天预测.png`、`分析/预测报告.txt`、`分析/未来30天预测.csv`

## 数据爬取

### 环境准备
- `python -m pip install -r requirements.txt`
- `python -m playwright install chromium`

### 使用方法
- 基本用法（指定币种与输出文件）：
  - `python scrape_cmb_rates_playwright.py --nbr 港币 --output hkd_rates.csv`
- 限制抓取页数（例如抓取前 20 页）：
  - `python scrape_cmb_rates_playwright.py --nbr 港币 --max_pages 20 --output hkd_rates_20pages.csv`
- 抓取所有页（不限制页数）：
  - `python scrape_cmb_rates_playwright.py --nbr 港币 --output hkd_rates_all.csv`
- 更换币种示例：
  - `python scrape_cmb_rates_playwright.py --nbr 美元 --max_pages 10 --output usd_rates_10pages.csv`
  - `python scrape_cmb_rates_playwright.py --nbr 欧元 --output eur_rates_all.csv`

### 参数说明
- `--nbr`：币种中文名（如 `港币`、`美元`、`欧元` 等）
- `--max_pages`：最多抓取页数；省略或设为非正数表示抓到最后一页
- `--output`：输出 CSV 文件路径

### CSV 字段
- `日期`、`汇买价`、`钞买价`、`汇卖价`、`钞卖价`

### 备注
- 页面为单页应用（SPA），脚本通过 Playwright 渲染页面后解析表格。
- 分页策略：优先使用“前往 + GO”跳转；失败时回退点击“下一页”链接。
- 更多示例命令请查看 `commands.txt`。

## 分析预测

### 方法与代码结构

#### 方法概述
- 多特征输入：引入周期特征（周内/月底的 `sin/cos`）、统计特征（`delta1` 一日差分、`ma7_diff` 七日均值偏差、`std7` 七日标准差）、周末指示（`is_weekend`），帮助模型理解节律与波动。
- 直接多步 LSTM（Seq2Seq 风格）：一次训练，直接输出未来 30 天，避免单步递归的误差累积，更好学习跨步相关性。
- 与普通 LSTM 的区别：普通做法多为单特征、单步训练 + 递归预测，曲线容易“趋于均值”，跨步相关性弱；本方法特征更丰富、训练目标直接对应未来序列，预测更具真实波动与长期一致性。

#### 代码结构（分析/fit_forecast_lstm.py）
- `_read_csv_safely`：多编码安全读取 CSV。
- `_ensure_chinese_font`：配置中文字体、负号显示。
- `load_series`：解析日期、清洗数值、按日期索引返回目标序列。
- `as_daily_continuous`：补齐为按日连续时间序列并前向填充。
- `build_features`：构造多特征矩阵（周期、统计、周末指示），首列为目标 `y`。
- `scale_features`：对 `delta1`/`ma7_diff`/`std7` 做列级 z-score。
- `scale_series` / `inverse_scale`：目标序列 min-max 缩放与反缩放。
- `make_supervised_multistep`：生成监督样本 `X:(n, lookback, n_features)` 与 `y:(n, horizon)`。
- `build_model`：LSTM(64, `return_sequences=True`) → LSTM(32) → Dense(horizon)。
- `_series_stats`：基础统计（起终点、极值、波动、连续涨跌）。
- `_build_trend_text`：将统计转为中文趋势文字。
- `evaluate_walk_forward`：一次训练、最近窗口直接多步评估（RMSE/MAE/MAPE）。
- `forecast_future`：使用全部样本滑窗训练，直接输出未来 30 天（缩放空间）。
- `plot_png`：静态趋势图（近 15 天实际 + 未来 30 天预测）。
- `write_html_chart`：交互趋势图（Chart.js，可缩放/悬浮查看）。
- `write_report_v2`：完整预测与方法说明报告（utf-8-sig，记事本友好）。
- `main`：端到端流程（读取→特征→评估→预测→图表与报告 → CSV）。

### 端到端流程
- 读取并按当前日期截断数据 → 构建特征与缩放目标。
- 评估最近窗口（直接多步） → 训练并预测未来 30 天。
- 输出静态 PNG、交互 HTML、报告 TXT、预测 CSV。

### 运行预测
- 安装依赖：`python -m pip install -r requirements.txt`
- 运行脚本（默认输入为 `hkd_rates_30pages.csv`）：
  - `python 分析/fit_forecast_lstm.py --rate_col "汇卖价" --lookback 30`
- 可选参数：
  - `--input`：输入 CSV（默认 `hkd_rates_30pages.csv`）
  - `--outdir`：输出目录（默认 `分析`）
  - `--rate_col`：目标列（默认 `汇卖价`）
  - `--current_date`：评估/预测的当前日期（默认使用数据最大日期），示例 `2025-11-03`
  - `--lookback`：回看窗口天数（默认 30）

### 预览与服务器
- 在浏览器打开交互图与报告前，请先启动本地服务器：
  - `python -m http.server 8000`
  - 若 8000 端口被占用：`python -m http.server 8080`
- 打开链接（8000 端口示例）：
  - 交互图：`http://localhost:8000/分析/未来30天预测.html`
  - 静态图：`http://localhost:8000/分析/未来30天预测.png`
  - 预测报告：`http://localhost:8000/分析/预测报告.txt`

### 输出文件索引
- `分析/未来30天预测.html`：交互趋势图
- `分析/未来30天预测.png`：静态趋势图
- `分析/预测报告.txt`：预测与方法说明报告
- `分析/未来30天预测.csv`：未来 30 天预测值（日期、预测值）

### 复现命令（示例）
- `python 分析/fit_forecast_lstm.py --rate_col "汇卖价" --lookback 30`

### 历史趋势图（可选）
- 使用 `分析/trend_plot.py` 仅生成历史趋势图：
  - `python 分析/trend_plot.py --input hkd_rates_30pages.csv --rate_col "汇卖价" --outdir 分析`