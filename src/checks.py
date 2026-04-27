from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd


@dataclass
class DQResult:
    rule_id: str
    dataset: str
    description: str
    passed: bool
    value: float | int
    threshold: float | int | None = None
    severity: str = "HIGH"

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "dataset": self.dataset,
            "description": self.description,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "severity": self.severity,
        }


def null_check(
    df: pd.DataFrame,
    col: str,
    rule_id: str,
    dataset: str,
    description: str,
    severity: str = "HIGH",
) -> DQResult:
    null_count = int(df[col].isna().sum()) if col in df.columns else -1
    return DQResult(
        rule_id=rule_id,
        dataset=dataset,
        description=description,
        passed=null_count == 0,
        value=null_count,
        threshold=0,
        severity=severity,
    )


def dup_check(
    df: pd.DataFrame,
    keys: Iterable[str],
    rule_id: str,
    dataset: str,
    description: str,
    severity: str = "HIGH",
) -> DQResult:
    keys_list = list(keys)
    if not all(k in df.columns for k in keys_list):
        dup_count = -1
    else:
        dup_count = int(df.duplicated(subset=keys_list).sum())

    return DQResult(
        rule_id=rule_id,
        dataset=dataset,
        description=description,
        passed=dup_count == 0,
        value=dup_count,
        threshold=0,
        severity=severity,
    )


def coverage_check(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_key: str,
    threshold_pct: float,
    rule_id: str,
    dataset: str,
    description: str,
    severity: str = "HIGH",
) -> DQResult:
    if join_key not in left_df.columns or join_key not in right_df.columns:
        coverage_pct = -1
    else:
        left_keys = set(left_df[join_key].dropna().unique())
        right_keys = set(right_df[join_key].dropna().unique())
        covered = len(left_keys & right_keys)
        total = len(left_keys)
        coverage_pct = (covered / total * 100) if total > 0 else 0

    passed = coverage_pct >= threshold_pct if coverage_pct >= 0 else False

    return DQResult(
        rule_id=rule_id,
        dataset=dataset,
        description=description,
        passed=passed,
        value=coverage_pct,
        threshold=threshold_pct,
        severity=severity,
    )


def normalize_id_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.upper()
    return df


def normalize_text_col(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


def black_friday_date(year: int) -> date:
    d = date(year, 11, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
    fourth_thu = first_thu + timedelta(weeks=3)
    return fourth_thu + timedelta(days=1)
