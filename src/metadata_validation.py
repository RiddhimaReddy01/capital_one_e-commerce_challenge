from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io_utils import write_json
from .config import Config


def validate_metadata_registry(
    settings: Config,
    buyers: pd.DataFrame,
    final_line: pd.DataFrame,
    order_level: pd.DataFrame,
) -> dict:
    registry_path = Path(settings.metadata_registry_path)
    if not registry_path.exists():
        report = {
            "passed": False,
            "error": f"Missing metadata registry: {registry_path}",
            "checks": [],
        }
        write_json(report, settings.reports_dir / "metadata_validation_report.json")
        return report

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    fields = data.get("fields", [])
    checks: list[dict] = []

    seen_names = set()
    duplicate_names = 0
    valid_grains = {"buyer", "line", "order"}
    grain_df = {"buyer": buyers, "line": final_line, "order": order_level}
    allowed_input_df = {
        "buyer": [buyers],
        "line": [final_line],
        "order": [order_level, final_line],  # allow upstream lineage inputs from line grain
    }

    required_keys = {"name", "grain", "definition", "formula", "inputs"}
    missing_required_entries = 0
    invalid_grains = 0
    missing_field_in_target = 0
    missing_input_columns = 0

    for f in fields:
        if not required_keys.issubset(set(f.keys())):
            missing_required_entries += 1
            continue
        name = f["name"]
        grain = f["grain"]
        inputs = f.get("inputs", [])

        if name in seen_names:
            duplicate_names += 1
        seen_names.add(name)

        if grain not in valid_grains:
            invalid_grains += 1
            continue

        target = grain_df[grain]
        if name not in target.columns:
            missing_field_in_target += 1

        for col in inputs:
            if not any(col in df.columns for df in allowed_input_df[grain]):
                missing_input_columns += 1

    checks.append(
        {
            "rule_id": "MD001",
            "description": "All metadata entries must contain required keys",
            "value": missing_required_entries,
            "passed": missing_required_entries == 0,
        }
    )
    checks.append(
        {
            "rule_id": "MD002",
            "description": "Metadata field names must be unique",
            "value": duplicate_names,
            "passed": duplicate_names == 0,
        }
    )
    checks.append(
        {
            "rule_id": "MD003",
            "description": "Metadata grain must be one of {buyer,line,order}",
            "value": invalid_grains,
            "passed": invalid_grains == 0,
        }
    )
    checks.append(
        {
            "rule_id": "MD004",
            "description": "Each metadata field must exist in the grain target dataset",
            "value": missing_field_in_target,
            "passed": missing_field_in_target == 0,
        }
    )
    checks.append(
        {
            "rule_id": "MD005",
            "description": "Metadata input columns must exist in the grain target dataset",
            "value": missing_input_columns,
            "passed": missing_input_columns == 0,
        }
    )

    passed = all(c["passed"] for c in checks)
    report = {
        "passed": passed,
        "registry_path": str(registry_path),
        "field_count": len(fields),
        "checks": checks,
    }
    write_json(report, settings.reports_dir / "metadata_validation_report.json")
    return report
