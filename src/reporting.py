from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config


def _report_path(config: Config, filename: str) -> Path:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    return config.reports_dir / filename


def write_processing_log(run_data: dict[str, Any], config: Config) -> None:
    path = _report_path(config, "processing_log.json")
    path.write_text(json.dumps(run_data, indent=2, default=str), encoding="utf-8")


def write_metadata_fields(fields: dict[str, Any], config: Config) -> None:
    path = _report_path(config, "metadata_derived_fields.json")
    path.write_text(json.dumps(fields, indent=2, default=str), encoding="utf-8")


def write_executive_summary(metrics: dict[str, Any], config: Config) -> None:
    path = _report_path(config, "executive_summary.md")

    summary = "# Executive Summary\n\n"
    summary += "Data Pipeline Execution Report\n\n"
    summary += json.dumps(metrics, indent=2, default=str)
    path.write_text(summary, encoding="utf-8")
