from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parquet_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == "object" or pd.api.types.is_string_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].astype("string")
    return safe_df


def write_table(df: pd.DataFrame, base_path: Path) -> None:
    ensure_dir(base_path.parent)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    try:
        _parquet_safe_frame(df).to_parquet(base_path.with_suffix(".parquet"), index=False)
    except Exception as e:
        print(f"[write_table] WARNING: Parquet write failed with {type(e).__name__}: {e}")
        raise


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
