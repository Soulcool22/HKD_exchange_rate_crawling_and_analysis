from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_table_rows(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    key_headers = {"日期", "汇买价", "钞买价", "汇卖价", "钞卖价"}

    target = None
    for table in tables:
        headers: list[str] = []
        thead = table.find("thead")
        if thead:
            headers = [normalize_text(th.get_text()) for th in thead.find_all(["th", "td"])]
        else:
            first_tr = table.find("tr")
            if first_tr:
                headers = [normalize_text(cell.get_text()) for cell in first_tr.find_all(["th", "td"])]

        headers = [re.sub(r"（.*?）", "", head) for head in headers]
        if key_headers.issubset(set(headers)):
            target = table
            break

    rows: list[dict[str, Any]] = []
    if not target:
        return rows

    tbody = target.find("tbody") or target
    for tr in tbody.find_all("tr"):
        cells = [normalize_text(td.get_text()) for td in tr.find_all("td")]
        if len(cells) >= 5 and "日期" not in "".join(cells):
            rows.append(
                {
                    "日期": cells[0],
                    "汇买价": cells[1],
                    "钞买价": cells[2],
                    "汇卖价": cells[3],
                    "钞卖价": cells[4],
                }
            )
    return rows

