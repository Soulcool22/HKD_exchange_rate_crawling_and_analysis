import asyncio
import csv
import re
from urllib.parse import urlencode

from playwright.async_api import async_playwright


BASE_URL = "https://fx.cmbchina.com/hq/history"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_table_rows(html: str):
    # 查找包含关键表头的表格并提取行
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    key_headers = {"日期", "汇买价", "钞买价", "汇卖价", "钞卖价"}
    target = None
    for table in tables:
        headers = []
        thead = table.find("thead")
        if thead:
            for th in thead.find_all(["th", "td"]):
                headers.append(normalize(th.get_text()))
        else:
            first_tr = table.find("tr")
            if first_tr:
                for cell in first_tr.find_all(["th", "td"]):
                    headers.append(normalize(cell.get_text()))
        headers = [re.sub(r"（.*?）", "", h) for h in headers]
        if key_headers.issubset(set(headers)):
            target = table
            break
    rows = []
    if target:
        tbody = target.find("tbody") or target
        for tr in tbody.find_all("tr"):
            cells = [normalize(td.get_text()) for td in tr.find_all("td")]
            if len(cells) >= 5 and "日期" not in "".join(cells):
                rows.append({
                    "日期": cells[0],
                    "汇买价": cells[1],
                    "钞买价": cells[2],
                    "汇卖价": cells[3],
                    "钞卖价": cells[4],
                })
    return rows


async def click_next(page) -> bool:
    # 寻找各种“下一页/下页”控件并尝试点击（div/span/a/button皆可）
    candidates = [
        'text=下一页',
        'div.next',
        'a:has-text("下一页")',
        'button:has-text("下一页")',
        'a:has-text("下页")',
        'button:has-text("下页")',
        'a:has-text("下一页>")',
        'a:has-text("下一页»")',
    ]
    for sel in candidates:
        loc = page.locator(sel)
        if await loc.count() > 0:
            try:
                await loc.first.click()
                # 等待可能的异步数据更新
                await page.wait_for_timeout(800)
                await page.wait_for_load_state('networkidle')
                return True
            except Exception:
                continue
    return False


async def get_total_pages(page) -> int | None:
    try:
        text = await page.evaluate("() => document.body.innerText")
        m = re.search(r"页/(\d+)页", text)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


async def goto_page_via_input(page, page_no: int) -> bool:
    try:
        input_loc = page.locator('.paginations_input')
        btn_loc = page.locator('.paginations_btn')
        if await input_loc.count() == 0 or await btn_loc.count() == 0:
            return False
        # 输入页码并点击 GO
        await input_loc.first.click()
        await input_loc.first.fill(str(page_no))
        await btn_loc.first.click()
        await page.wait_for_timeout(800)
        await page.wait_for_load_state('networkidle')
        return True
    except Exception:
        return False


async def scrape_all_pages(nbr: str, max_pages: int | None = None) -> list[dict]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="zh-CN")
        page = await context.new_page()
        url = f"{BASE_URL}?" + urlencode({"nbr": nbr})
        await page.goto(url, wait_until="networkidle")
        # 等待渲染
        await page.wait_for_timeout(1000)

        all_rows: list[dict] = []
        seen_keys = set()
        pages = 0
        while True:
            html = await page.content()
            rows = extract_table_rows(html)
            # 去重并汇总
            for r in rows:
                key = (r.get("日期"), r.get("汇买价"), r.get("汇卖价"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_rows.append(r)
            pages += 1
            if max_pages and pages >= max_pages:
                break
            # 优先使用输入框+GO 翻页（更稳定），否则尝试点击“下一页”
            total_pages = await get_total_pages(page)
            if total_pages and pages < total_pages:
                changed = await goto_page_via_input(page, pages + 1)
                if not changed:
                    changed = await click_next(page)
            else:
                changed = await click_next(page)
            if not changed:
                break
        await browser.close()
        return all_rows


def save_csv(rows: list[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["日期", "汇买价", "钞买价", "汇卖价", "钞卖价"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="使用 Playwright 渲染抓取招商银行历史汇率，支持自动翻页")
    parser.add_argument("--nbr", default="港币", help="货币中文名，例如：港币、美元、欧元等")
    parser.add_argument("--output", default="cmb_rates_playwright.csv", help="输出CSV文件路径")
    parser.add_argument("--max_pages", type=int, default=None, help="最多抓取多少页，默认抓到最后一页")
    args = parser.parse_args()

    rows = asyncio.run(scrape_all_pages(args.nbr, max_pages=args.max_pages))
    print(f"抓取到 {len(rows)} 条")
    save_csv(rows, args.output)
    print(f"已保存到 {args.output}")