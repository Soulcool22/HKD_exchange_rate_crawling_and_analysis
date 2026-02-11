from __future__ import annotations

import asyncio
import re
from urllib.parse import urlencode

from playwright.async_api import async_playwright

from .parser import extract_table_rows


async def _click_next(page) -> bool:
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
    for selector in candidates:
        locator = page.locator(selector)
        if await locator.count() > 0:
            try:
                await locator.first.click()
                await page.wait_for_timeout(800)
                await page.wait_for_load_state("networkidle")
                return True
            except Exception:
                continue
    return False


async def _get_total_pages(page) -> int | None:
    try:
        text = await page.evaluate("() => document.body.innerText")
        matched = re.search(r"页/(\d+)页", text)
        if matched:
            return int(matched.group(1))
    except Exception:
        return None
    return None


async def _goto_page(page, page_no: int) -> bool:
    try:
        input_loc = page.locator(".paginations_input")
        btn_loc = page.locator(".paginations_btn")
        if await input_loc.count() == 0 or await btn_loc.count() == 0:
            return False
        await input_loc.first.click()
        await input_loc.first.fill(str(page_no))
        await btn_loc.first.click()
        await page.wait_for_timeout(800)
        await page.wait_for_load_state("networkidle")
        return True
    except Exception:
        return False


async def scrape_pages(base_url: str, currency: str, max_pages: int | None = None) -> list[dict]:
    async with async_playwright() as player:
        browser = await player.chromium.launch(headless=True)
        context = await browser.new_context(locale="zh-CN")
        page = await context.new_page()
        url = f"{base_url}?" + urlencode({"nbr": currency})
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(1000)

        all_rows: list[dict] = []
        seen = set()
        current_page = 0
        while True:
            html = await page.content()
            rows = extract_table_rows(html)
            for row in rows:
                key = (row.get("日期"), row.get("汇买价"), row.get("汇卖价"))
                if key not in seen:
                    seen.add(key)
                    all_rows.append(row)

            current_page += 1
            if max_pages and current_page >= max_pages:
                break

            total_pages = await _get_total_pages(page)
            if total_pages and current_page < total_pages:
                changed = await _goto_page(page, current_page + 1)
                if not changed:
                    changed = await _click_next(page)
            else:
                changed = await _click_next(page)

            if not changed:
                break

        await browser.close()
    return all_rows


def scrape_pages_sync(base_url: str, currency: str, max_pages: int | None = None) -> list[dict]:
    return asyncio.run(scrape_pages(base_url=base_url, currency=currency, max_pages=max_pages))

