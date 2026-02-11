from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hkd_fx.config import load_config, resolve_path
from hkd_fx.dataset.canonical_store import (
    load_master_rates,
    merge_master,
    normalize_crawl_rows,
    save_master_rates,
)


def _read_csv_fallback(path: Path, encodings: tuple[str, ...]) -> pd.DataFrame:
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as error:
            last_error = error
            continue
    raise last_error


def main() -> None:
    config = load_config()
    root = resolve_path(".")

    legacy_files = [
        root / "hkd_rates_30pages.csv",
        root / "actual_rates_recent.csv",
    ]

    master = load_master_rates(config, root)
    merged = master.copy()

    for path in legacy_files:
        if not path.exists():
            continue
        df = _read_csv_fallback(path, config.encoding_fallbacks)
        incoming = normalize_crawl_rows(df, currency=config.currency, source_run_id=f"legacy_{path.stem}")
        merged = merge_master(merged, incoming)
        print(f"merged legacy file: {path.name}, rows={len(incoming)}")

    output = save_master_rates(config, root, merged)
    print(f"master saved: {output}")
    print(f"master rows: {len(merged)}")


if __name__ == "__main__":
    main()
