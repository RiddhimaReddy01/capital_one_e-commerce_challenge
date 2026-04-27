from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import ensure_dir, write_json, write_table


def _series_distribution(series: pd.Series, top_n: int = 20) -> list[dict[str, Any]]:
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False).head(top_n)
    total = int(len(series))
    out: list[dict[str, Any]] = []
    for value, count in counts.items():
        out.append(
            {
                "value": None if value == "<NA>" else str(value),
                "count": int(count),
                "pct": round((int(count) / total * 100.0), 4) if total else 0.0,
            }
        )
    return out


def _null_profile(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in df.columns}


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            nxt = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, nxt))
    elif isinstance(obj, list):
        return out
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def _write_buyers_eda(buyers: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    payload = {
        "rows": int(len(buyers)),
        "duplicate_buyer_id_count": int(buyers.duplicated(subset=["buyer_id"]).sum()) if "buyer_id" in buyers.columns else -1,
        "null_profile": _null_profile(buyers),
        "invalid_referral_count": int(buyers["is_referred_invalid"].sum()) if "is_referred_invalid" in buyers.columns else 0,
        "distributions": {},
    }

    for col in ["customer_group", "customer_segment", "region", "preferred_channel", "is_referred"]:
        if col in buyers.columns:
            payload["distributions"][col] = _series_distribution(buyers[col], top_n=25)

    write_json(payload, out_dir / "buyers_eda_summary.json")
    return payload


def _write_sales_eda(sales: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    working = sales.copy()
    if "order_datetime" in working.columns:
        working["order_datetime"] = pd.to_datetime(working["order_datetime"], errors="coerce")

    status_mix: list[dict[str, Any]] = []
    if "order_status" in working.columns:
        status_mix = _series_distribution(working["order_status"], top_n=25)

    quantity_summary: dict[str, Any] = {}
    if "quantity" in working.columns:
        qty = pd.to_numeric(working["quantity"], errors="coerce")
        quantity_summary = {
            "count": int(qty.notna().sum()),
            "min": float(qty.min()) if qty.notna().any() else None,
            "p25": float(qty.quantile(0.25)) if qty.notna().any() else None,
            "median": float(qty.quantile(0.5)) if qty.notna().any() else None,
            "p75": float(qty.quantile(0.75)) if qty.notna().any() else None,
            "max": float(qty.max()) if qty.notna().any() else None,
        }
    quantity_original_summary: dict[str, Any] = {}
    if "quantity_original" in working.columns:
        qty0 = pd.to_numeric(working["quantity_original"], errors="coerce")
        quantity_original_summary = {
            "count": int(qty0.notna().sum()),
            "min": float(qty0.min()) if qty0.notna().any() else None,
            "p25": float(qty0.quantile(0.25)) if qty0.notna().any() else None,
            "median": float(qty0.quantile(0.5)) if qty0.notna().any() else None,
            "p75": float(qty0.quantile(0.75)) if qty0.notna().any() else None,
            "max": float(qty0.max()) if qty0.notna().any() else None,
        }

    monthly = pd.DataFrame(columns=["year_month", "order_count", "line_count"])
    if {"order_id", "order_datetime"}.issubset(working.columns):
        monthly = (
            working.dropna(subset=["order_datetime"])
            .assign(year_month=lambda d: d["order_datetime"].dt.to_period("M").astype(str))
            .groupby("year_month", as_index=False)
            .agg(order_count=("order_id", "nunique"), line_count=("order_id", "size"))
            .sort_values("year_month")
            .reset_index(drop=True)
        )
        write_table(monthly, out_dir / "sales_eda_monthly_trend")

    top_skus = pd.DataFrame(columns=["sku_id", "line_count", "order_count", "total_quantity"])
    if {"sku_id", "order_id", "quantity"}.issubset(working.columns):
        top_skus = (
            working.groupby("sku_id", as_index=False)
            .agg(
                line_count=("order_id", "size"),
                order_count=("order_id", "nunique"),
                total_quantity=("quantity", "sum"),
            )
            .sort_values(["line_count", "order_count"], ascending=False)
            .head(100)
            .reset_index(drop=True)
        )
        write_table(top_skus, out_dir / "sales_eda_top_skus")

    payload = {
        "rows": int(len(working)),
        "duplicate_line_grain_count": int(working.duplicated(subset=["order_id", "buyer_id", "sku_id"]).sum())
        if {"order_id", "buyer_id", "sku_id"}.issubset(working.columns)
        else -1,
        "null_profile": _null_profile(working),
        "status_mix": status_mix,
        "quantity_summary": quantity_summary,
        "quantity_original_summary": quantity_original_summary,
        "cleaning_audit": dict(working.attrs.get("cleaning_audit", {})),
        "monthly_trend_rows": int(len(monthly)),
        "top_sku_rows": int(len(top_skus)),
    }
    write_json(payload, out_dir / "sales_eda_summary.json")
    return payload


def _write_products_eda(products: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    working = products.copy()
    duplicate_sku_count = int(working.duplicated(subset=["sku_id"]).sum()) if "sku_id" in working.columns else -1

    numeric_bounds: dict[str, Any] = {}
    for col in ["price", "margin", "current_stock"]:
        if col in working.columns:
            ser = pd.to_numeric(working[col], errors="coerce")
            numeric_bounds[col] = {
                "null_count": int(ser.isna().sum()),
                "min": float(ser.min()) if ser.notna().any() else None,
                "max": float(ser.max()) if ser.notna().any() else None,
                "mean": float(ser.mean()) if ser.notna().any() else None,
            }

    category_breakdown = pd.DataFrame(columns=["category_name", "sku_count"])
    if "category_name" in working.columns:
        category_breakdown = (
            working.groupby("category_name", as_index=False)
            .agg(sku_count=("sku_id", "nunique"))
            .sort_values("sku_count", ascending=False)
            .reset_index(drop=True)
        )
        write_table(category_breakdown, out_dir / "products_eda_category_breakdown")

    payload = {
        "rows": int(len(working)),
        "duplicate_sku_id_count": duplicate_sku_count,
        "null_profile": _null_profile(working),
        "margin_imputed_count": int(working["margin_imputed"].sum()) if "margin_imputed" in working.columns else 0,
        "vendor_product_conflict_count": int(working["vendor_product_conflict"].sum())
        if "vendor_product_conflict" in working.columns
        else 0,
        "inactive_with_stock_count": int(working["inactive_with_stock_flag"].sum())
        if "inactive_with_stock_flag" in working.columns
        else 0,
        "numeric_bounds": numeric_bounds,
        "category_breakdown_rows": int(len(category_breakdown)),
    }

    write_json(payload, out_dir / "products_eda_summary.json")
    return payload


def _write_final_line_eda(final_line: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    payload = {
        "rows": int(len(final_line)),
        "duplicate_order_buyer_sku_count": int(final_line.duplicated(subset=["order_id", "buyer_id", "sku_id"]).sum())
        if {"order_id", "buyer_id", "sku_id"}.issubset(final_line.columns)
        else -1,
        "null_profile": _null_profile(final_line),
    }
    write_json(payload, out_dir / "final_line_eda_summary.json")
    return payload


def _write_order_level_eda(order_level: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    working = order_level.copy()
    if "order_datetime_company" in working.columns:
        working["order_datetime_company"] = pd.to_datetime(working["order_datetime_company"], errors="coerce")

    monthly = pd.DataFrame(columns=["year_month", "order_count", "total_revenue", "total_profit"])
    if {"order_id", "order_datetime_company", "order_revenue", "order_profit"}.issubset(working.columns):
        monthly = (
            working.dropna(subset=["order_datetime_company"])
            .assign(
                year_month=lambda d: (
                    d["order_datetime_company"]
                    .dt.tz_localize(None)
                    .dt.to_period("M")
                    .astype(str)
                )
            )
            .groupby("year_month", as_index=False)
            .agg(
                order_count=("order_id", "nunique"),
                total_revenue=("order_revenue", "sum"),
                total_profit=("order_profit", "sum"),
            )
            .sort_values("year_month")
            .reset_index(drop=True)
        )
        write_table(monthly, out_dir / "order_level_eda_monthly_trend")

    profit_summary: dict[str, Any] = {}
    if "order_profit" in working.columns:
        p = pd.to_numeric(working["order_profit"], errors="coerce")
        profit_summary = {
            "count": int(p.notna().sum()),
            "min": float(p.min()) if p.notna().any() else None,
            "p25": float(p.quantile(0.25)) if p.notna().any() else None,
            "median": float(p.quantile(0.5)) if p.notna().any() else None,
            "p75": float(p.quantile(0.75)) if p.notna().any() else None,
            "max": float(p.max()) if p.notna().any() else None,
            "negative_order_profit_count": int((p < 0).sum()) if p.notna().any() else 0,
        }

    payload = {
        "rows": int(len(working)),
        "duplicate_order_id_count": int(working.duplicated(subset=["order_id"]).sum()) if "order_id" in working.columns else -1,
        "null_profile": _null_profile(working),
        "order_profit_summary": profit_summary,
        "monthly_trend_rows": int(len(monthly)),
    }
    write_json(payload, out_dir / "order_level_eda_summary.json")
    return payload


def build_eda_artifacts(
    buyers: pd.DataFrame,
    sales: pd.DataFrame,
    products: pd.DataFrame,
    processed_dir: Path,
    final_line: pd.DataFrame | None = None,
    order_level: pd.DataFrame | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(processed_dir / "eda")
    previous_summary = _read_json_if_exists(out_dir / "eda_summary.json")

    buyers_summary = _write_buyers_eda(buyers, out_dir)
    sales_summary = _write_sales_eda(sales, out_dir)
    products_summary = _write_products_eda(products, out_dir)

    summary = {
        "buyers": buyers_summary,
        "sales": sales_summary,
        "products": products_summary,
    }
    if final_line is not None:
        summary["final_line"] = _write_final_line_eda(final_line, out_dir)
    if order_level is not None:
        summary["order_level"] = _write_order_level_eda(order_level, out_dir)

    write_json(summary, out_dir / "eda_summary.json")

    drift_report: dict[str, Any] = {"has_previous": previous_summary is not None, "metric_deltas": []}
    if previous_summary is not None:
        curr_flat = _flatten_numeric(summary)
        prev_flat = _flatten_numeric(previous_summary)
        shared_keys = sorted(set(curr_flat).intersection(set(prev_flat)))
        deltas: list[dict[str, Any]] = []
        for key in shared_keys:
            prev_val = prev_flat[key]
            curr_val = curr_flat[key]
            delta = curr_val - prev_val
            pct_change = (delta / abs(prev_val) * 100.0) if prev_val != 0 else None
            deltas.append(
                {
                    "metric": key,
                    "previous": prev_val,
                    "current": curr_val,
                    "delta": delta,
                    "pct_change": pct_change,
                }
            )
        drift_report["metric_deltas"] = deltas
    write_json(drift_report, out_dir / "eda_drift_report.json")
    summary["drift_report"] = {"path": str(out_dir / "eda_drift_report.json")}
    return summary
