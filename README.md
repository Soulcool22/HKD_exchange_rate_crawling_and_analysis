# 招商银行历史汇率抓取（通用说明）

此工作区包含：
- `scrape_cmb_rates_playwright.py`：Playwright 渲染并自动翻页抓取脚本
- `requirements.txt`：依赖清单
- `commands.txt`：常用安装与运行命令便于复制

## 环境准备
- `python -m pip install -r requirements.txt`
- `python -m playwright install chromium`

## 使用方法
- 基本用法（指定币种与输出文件）：
  - `python scrape_cmb_rates_playwright.py --nbr 港币 --output hkd_rates.csv`
- 限制抓取页数（例如抓取前 20 页）：
  - `python scrape_cmb_rates_playwright.py --nbr 港币 --max_pages 20 --output hkd_rates_20pages.csv`
- 抓取所有页（不限制页数）：
  - `python scrape_cmb_rates_playwright.py --nbr 港币 --output hkd_rates_all.csv`
- 更换币种示例：
  - `python scrape_cmb_rates_playwright.py --nbr 美元 --max_pages 10 --output usd_rates_10pages.csv`
  - `python scrape_cmb_rates_playwright.py --nbr 欧元 --output eur_rates_all.csv`

## 参数说明
- `--nbr`：币种中文名（如 `港币`、`美元`、`欧元` 等）
- `--max_pages`：最多抓取页数；省略或设为非正数表示抓到最后一页
- `--output`：输出 CSV 文件路径

## CSV 字段
- `日期`、`汇买价`、`钞买价`、`汇卖价`、`钞卖价`

## 备注
- 页面为单页应用（SPA），脚本通过 Playwright 渲染页面后解析表格。
- 分页策略：优先使用“前往 + GO”跳转；失败时回退点击“下一页”链接。
- 更多示例命令请查看 `commands.txt`。