from __future__ import annotations

from pathlib import Path

import yaml

from .contracts import PipelineConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _project_root() / path


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    path = resolve_path(config_path or "config/default.yaml")
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    pipe = raw["pipeline"]
    paths = raw["paths"]
    return PipelineConfig(
        currency=pipe["currency"],
        rate_col=pipe["rate_col"],
        crawl_pages=int(pipe["crawl_pages"]),
        lookback=int(pipe["lookback"]),
        horizon=int(pipe["horizon"]),
        history_window_days=int(pipe["history_window_days"]),
        model_name=pipe["model_name"],
        source_url=pipe["source_url"],
        data_dir=paths["data_dir"],
        artifacts_dir=paths["artifacts_dir"],
        analysis_mirror_dir=paths["analysis_mirror_dir"],
        master_rates_file=paths["master_rates_file"],
        forecast_registry_file=paths["forecast_registry_file"],
        comparison_registry_file=paths["comparison_registry_file"],
        crawl_snapshot_dir=paths["crawl_snapshot_dir"],
        encoding_fallbacks=tuple(raw.get("encoding_fallbacks", ["utf-8-sig", "utf-8", "gbk", "gb18030"])),
    )

