from __future__ import annotations

import pandas as pd

from .checks import DQResult, null_check, dup_check, coverage_check
from .io_utils import write_json
from .config import Config


def check_buyers_clean(df: pd.DataFrame) -> list[DQResult]:
    results = []
    results.append(
        null_check(df, "buyer_id", "DQ_B001", "buyers_clean", "buyer_id null count should be 0")
    )
    results.append(
        dup_check(df, ["buyer_id"], "DQ_B002", "buyers_clean", "buyer_id duplicate count should be 0")
    )

    return results


def check_sales_clean(df: pd.DataFrame) -> list[DQResult]:
    results = []
    results.append(
        null_check(df, "order_id", "DQ_S001", "sales_clean", "order_id null count should be 0")
    )
    results.append(
        null_check(df, "buyer_id", "DQ_S002", "sales_clean", "buyer_id null count should be 0")
    )
    results.append(
        null_check(df, "sku_id", "DQ_S003", "sales_clean", "sku_id null count should be 0")
    )
    if "order_status" in df.columns:
        results.append(
            null_check(
                df,
                "order_status",
                "DQ_S004",
                "sales_clean",
                "order_status null count should be 0",
            )
        )

    if "quantity" in df.columns:
        qty = pd.to_numeric(df["quantity"], errors="coerce")
        qty_null = int(qty.isna().sum())
        qty_nonpositive = int((qty <= 0).sum()) if qty.notna().any() else 0

        results.append(
            DQResult(
                "DQ_S005",
                "sales_clean",
                "quantity null count should be 0 after cleaning",
                qty_null == 0,
                qty_null,
                0,
            )
        )
        results.append(
            DQResult(
                "DQ_S006",
                "sales_clean",
                "quantity <= 0 count should be 0 after cleaning",
                qty_nonpositive == 0,
                qty_nonpositive,
                0,
            )
        )

    if "quantity_was_capped" in df.columns:
        capped_count = int(pd.Series(df["quantity_was_capped"]).fillna(False).astype(bool).sum())
        results.append(
            DQResult(
                "DQ_S007",
                "sales_clean",
                "quantity capped row count (informational)",
                True,
                capped_count,
                severity="INFO",
            )
        )

    # DQ_S008: quantity cap enforcement
    if {"quantity", "quantity_cap_value"}.issubset(df.columns):
        qty = pd.to_numeric(df["quantity"], errors="coerce")
        cap = pd.to_numeric(df["quantity_cap_value"], errors="coerce")
        qty_exceeds_cap = int((qty > cap).sum())
        results.append(
            DQResult(
                "DQ_S008",
                "sales_clean",
                "quantity above cap count should be 0 after capping",
                qty_exceeds_cap == 0,
                qty_exceeds_cap,
                0,
            )
        )

    return results


def check_products_clean(df: pd.DataFrame) -> list[DQResult]:
    results = []

    results.append(
        dup_check(df, ["sku_id"], "DQ_P001", "products_clean", "sku_id duplicate count should be 0")
    )

    if "price" in df.columns:
        price_nulls = int(df["price"].isna().sum())
        price_negative = int((pd.to_numeric(df["price"], errors="coerce") < 0).sum())

        results.append(
            DQResult(
                "DQ_P002",
                "products_clean",
                "price null count (informational)",
                True,
                price_nulls,
                severity="INFO",
            )
        )
        results.append(
            DQResult(
                "DQ_P003",
                "products_clean",
                "price negative count should be 0",
                price_negative == 0,
                price_negative,
                0,
            )
        )

    if "margin" in df.columns:
        margin_series = pd.to_numeric(df["margin"], errors="coerce")
        margin_lt_neg1 = int((margin_series < -1).sum())
        margin_gt_1 = int((margin_series > 1).sum())

        results.append(
            DQResult(
                "DQ_P004",
                "products_clean",
                "margin < -1 count should be 0",
                margin_lt_neg1 == 0,
                margin_lt_neg1,
                0,
                severity="MEDIUM",
            )
        )
        results.append(
            DQResult(
                "DQ_P005",
                "products_clean",
                "margin > 1 count should be 0",
                margin_gt_1 == 0,
                margin_gt_1,
                0,
                severity="MEDIUM",
            )
        )

    return results


def check_final_line(df: pd.DataFrame, sales: pd.DataFrame, settings: Config) -> list[DQResult]:
    results = []

    # DQ_J001: Join coverage
    results.append(
        coverage_check(
            sales,
            df,
            "buyer_id",
            settings.min_join_coverage_pct,
            "DQ_J001",
            "final_dataset_line",
            "join coverage pct should be above minimum threshold",
            severity="MEDIUM",
        )
    )

    # DQ_J002: Missing IDs (informational)
    sales_ids = set(sales["buyer_id"].astype("string").str.strip().dropna().unique())
    final_ids = set(df["buyer_id"].astype("string").str.strip().dropna().unique())
    missing_ids = sales_ids - final_ids

    results.append(
        DQResult(
            "DQ_J002",
            "final_dataset_line",
            "missing buyer IDs from sales after join (informational)",
            True,
            len(missing_ids),
            severity="INFO",
        )
    )

    return results


def check_order_level(df: pd.DataFrame) -> list[DQResult]:
    results = []

    # DQ_O001: order_id uniqueness
    results.append(
        dup_check(
            df,
            ["order_id"],
            "DQ_O001",
            "order_level",
            "order_id duplicate count should be 0 at order grain",
        )
    )

    # DQ_O005: invalid year values
    if "year" in df.columns:
        invalid_years = int(((df["year"] < 2020) | (df["year"] > 2026)).sum())
        results.append(
            DQResult(
                "DQ_O005",
                "order_level",
                "invalid year values (< 2020 or > 2026) should be filtered",
                invalid_years == 0,
                invalid_years,
                0,
                severity="HIGH",
            )
        )

    # DQ_O002: negative shipping fees
    if "customer_shipping_fee" in df.columns:
        ship_negative = int((pd.to_numeric(df["customer_shipping_fee"], errors="coerce") < 0).sum())
        results.append(
            DQResult(
                "DQ_O002",
                "order_level",
                "customer shipping fee negative count should be 0",
                ship_negative == 0,
                ship_negative,
                0,
            )
        )

    # DQ_O003: negative revenue (informational)
    if "order_revenue" in df.columns:
        revenue_negative = int((pd.to_numeric(df["order_revenue"], errors="coerce") < 0).sum())
        results.append(
            DQResult(
                "DQ_O003",
                "order_level",
                "order_revenue negative count (informational)",
                True,
                revenue_negative,
                severity="INFO",
            )
        )

    # DQ_O004: negative profit (material data quality issue)
    if "order_profit" in df.columns:
        profit_negative = int((pd.to_numeric(df["order_profit"], errors="coerce") < 0).sum())
        results.append(
            DQResult(
                "DQ_O004",
                "order_level",
                "order_profit < 0 count (should investigate for data quality)",
                True,
                profit_negative,
                severity="MEDIUM",
            )
        )

    return results


def run_quality_checks(
    buyers: pd.DataFrame,
    sales: pd.DataFrame,
    products: pd.DataFrame,
    final_line: pd.DataFrame,
    order_level: pd.DataFrame,
    settings: Config,
) -> dict:
    checks = []

    # Run per-dataset checks
    checks.extend(check_buyers_clean(buyers))
    checks.extend(check_sales_clean(sales))
    checks.extend(check_products_clean(products))
    checks.extend(check_final_line(final_line, sales, settings))
    checks.extend(check_order_level(order_level))

    # Determine pass/fail
    passed = all(c.passed for c in checks if c.severity in {"HIGH", "MEDIUM"})

    sales_ids = set(sales["buyer_id"].astype("string").str.strip().dropna().unique())
    final_ids = set(final_line["buyer_id"].astype("string").str.strip().dropna().unique())
    missing_ids = sales_ids - final_ids
    covered_ids = sales_ids & final_ids

    # Coverage semantics:
    # - coverage_pct: join-key (buyer_id) coverage, aligned with coverage_check logic.
    # - row_retention_pct: row-level retention after joins/filters.
    coverage_pct = float(len(covered_ids) / len(sales_ids) * 100) if len(sales_ids) else 0.0
    row_retention_pct = float(len(final_line) / len(sales) * 100) if len(sales) else 0.0

    result = {
        "passed": passed,
        "summary": {
            "buyers_rows": int(len(buyers)),
            "sales_rows": int(len(sales)),
            "products_rows": int(len(products)),
            "final_line_rows": int(len(final_line)),
            "order_level_rows": int(len(order_level)),
            "coverage_pct": round(coverage_pct, 2),
            "row_retention_pct": round(row_retention_pct, 2),
            "missing_sales_buyer_ids_in_final": int(len(missing_ids)),
        },
        "checks": [c.to_dict() for c in checks],
    }

    write_json(result, settings.reports_dir / "data_quality_report.json")
    return result
