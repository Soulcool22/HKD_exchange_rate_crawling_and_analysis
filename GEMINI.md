# 项目概述

这是一个结合了网络爬虫与数据分析的 Python 项目，旨在从招商银行网站抓取指定货币（如港币）的历史汇率数据，并利用长短期记忆网络（LSTM）模型对未来30天的汇率走势进行预测。

项目分为两个核心模块：
1.  **数据抓取模块**: 使用 `Playwright` 模拟浏览器行为，实现动态网页内容的渲染和自动翻页，从而抓取完整的历史汇率表格，并将结果保存为 CSV 文件。
2.  **分析预测模块**: 读取抓取到的数据，通过 `pandas` 进行清洗和预处理。随后，构建包括周期特征、统计特征在内的多维特征矩阵，训练一个直接多步输出（Seq2Seq风格）的 LSTM 模型。最终生成对未来30天的预测结果，并输出多种格式的报告。

## 技术栈

- **数据抓取**: Python, Playwright, BeautifulSoup4
- **数据处理**: pandas, numpy
- **模型训练**: TensorFlow/Keras
- **数据可视化**: matplotlib, Chart.js (通过HTML模板)

## 运行与构建

### 1. 环境准备

首先，需要安装项目所需的 Python 依赖包，并安装 Playwright 的浏览器驱动。

```bash
# 安装 requirements.txt 中的依赖
python -m pip install -r requirements.txt

# 安装 Playwright 所需的 chromium 浏览器核心
python -m playwright install chromium
```

### 2. 数据抓取

运行 `scrape_cmb_rates_playwright.py` 脚本来抓取数据。你可以通过命令行参数指定货币名称和输出文件名。

```bash
# 示例：抓取港币的所有历史汇率数据
python scrape_cmb_rates_playwright.py --nbr 港币 --output hkd_rates.csv

# 示例：只抓取美元的前10页数据
python scrape_cmb_rates_playwright.py --nbr 美元 --max_pages 10 --output usd_rates_10pages.csv
```

### 3. 分析与预测

数据抓取完成后，运行 `分析/fit_forecast_lstm.py` 脚本进行模型训练和预测。脚本会使用 `hkd_rates_30pages.csv` 作为默认输入数据。

```bash
# 运行预测脚本，使用默认配置
python 分析/fit_forecast_lstm.py --rate_col "汇卖价" --lookback 30
```

脚本执行成功后，会在 `分析/` 目录下生成以下文件：
- `未来30天预测.png`: 静态趋势图。
- `未来30天预测.html`: 可交互的 HTML 趋势图。
- `未来30天预测.csv`: 包含未来30天预测值的 CSV 文件。
- `预测报告.txt`: 详细的预测方法说明、评估指标和趋势分析报告。

## 开发约定

- **代码风格**: 项目遵循 PEP 8 规范。代码注释非常详尽，尤其是在分析预测脚本中，每个函数都有详细的功能说明、目的和行为描述，便于理解和维护。
- **模块化**: 功能被清晰地划分到不同的脚本和函数中。例如，数据加载、特征工程、模型构建、评估和报告生成都有各自独立的函数。
- **错误处理**: 代码中包含对文件编码、网络请求等潜在问题的处理，例如 `_read_csv_safely` 函数会尝试多种编码格式读取文件。
- **可复现性**: 预测脚本通过设置固定的随机种子 (`tf.random.set_seed(42)`, `np.random.seed(42)`) 来保证模型训练和预测结果的可复现性。
- **配置文件**: 核心参数（如回看窗口 `lookback`、目标列 `rate_col`）通过 `argparse` 进行管理，方便从命令行调整。
