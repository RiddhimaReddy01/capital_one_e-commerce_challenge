from __future__ import annotations

import pandas as pd

from .checks import DQResult, null_check, dup_check, coverage_check
from .config import Config


def _col_present_check(rule_id: str, dataset: str, col: str) -> DQResult:
    return DQResult(
        rule_id=rule_id,
        dataset=dataset,
        description=f"{col} column present",
        passed=True,
        value=1,
        severity="INFO",
    )


def _required_cols_check(df: pd.DataFrame, cols: list[str], rule_id: str, dataset: str) -> DQResult:
    missing = [c for c in cols if c not in df.columns]
    return DQResult(
        rule_id=rule_id,
        dataset=dataset,
        description="all required columns present",
        passed=len(missing) == 0,
        value=len(missing),
        threshold=0,
        severity="CRITICAL",
    )


def validate_buyers_raw(df: pd.DataFrame) -> list[DQResult]:
    results = []
    results.append(
        null_check(df, "buyer_id", "RV_B001", "buyers_raw", "buyer_id has no nulls")
    )
    results.append(
        dup_check(df, ["buyer_id"], "RV_B002", "buyers_raw", "buyer_id is unique")
    )
    if "is_referred" in df.columns:
        invalid_referred = (
            df["is_referred"].notna()
            & ~df["is_referred"].isin([0, 1])
        ).sum()
        results.append(
            DQResult(
                rule_id="RV_B003",
                dataset="buyers_raw",
                description="is_referred values are 0, 1, or null",
                passed=invalid_referred == 0,
                value=invalid_referred,
                threshold=0,
                severity="MEDIUM",
            )
        )

    if "signup_date" in df.columns:
        results.append(_col_present_check("RV_B004", "buyers_raw", "signup_date"))

    return results


def validate_sales_raw(df: pd.DataFrame) -> list[DQResult]:
    results = []
    results.append(
        null_check(df, "order_id", "RV_S001", "sales_raw", "order_id has no nulls")
    )
    results.append(
        null_check(df, "buyer_id", "RV_S002", "sales_raw", "buyer_id has no nulls")
    )
    results.append(
        null_check(df, "sku_id", "RV_S003", "sales_raw", "sku_id has no nulls")
    )
    if "order_datetime" in df.columns:
        results.append(_col_present_check("RV_S004", "sales_raw", "order_datetime"))

    if "quantity" in df.columns:
        non_positive = (df["quantity"] <= 0).sum()
        results.append(
            DQResult(
                rule_id="RV_S005",
                dataset="sales_raw",
                description="quantity values > 0",
                passed=non_positive == 0,
                value=non_positive,
                threshold=0,
                severity="MEDIUM",
            )
        )

    results.append(
        _required_cols_check(
            df,
            ["order_id", "buyer_id", "sku_id", "order_datetime", "quantity"],
            "RV_S006",
            "sales_raw",
        )
    )

    return results


def validate_products_raw(df: pd.DataFrame) -> list[DQResult]:
    results = []
    if "price" in df.columns:
        results.append(_col_present_check("RV_P001", "products_raw", "price"))

    has_sku = "sku_id" in df.columns
    has_components = all(c in df.columns for c in ["subcategory_id", "vendor", "product_num"])
    sku_derivable_missing = 0 if (has_sku or has_components) else 1
    results.append(
        DQResult(
            rule_id="RV_P002",
            dataset="products_raw",
            description="products has sku_id or derivable sku components {subcategory_id,vendor,product_num}",
            passed=sku_derivable_missing == 0,
            value=sku_derivable_missing,
            threshold=0,
            severity="CRITICAL",
        )
    )

    results.append(
        _required_cols_check(df, ["price"], "RV_P003", "products_raw")
    )

    return results


def validate_cross_dataset(
    sales_raw: pd.DataFrame,
    buyers_raw: pd.DataFrame,
    products_raw: pd.DataFrame,
    settings: Config,
) -> list[DQResult]:
    results = []
    results.append(
        coverage_check(
            sales_raw,
            buyers_raw,
            "buyer_id",
            settings.min_join_coverage_pct,
            "RV_X001",
            "cross_dataset",
            f"sales.buyer_id coverage to buyers >= {settings.min_join_coverage_pct}%",
            severity="HIGH",
        )
    )
    results.append(
        coverage_check(
            sales_raw,
            products_raw,
            "sku_id",
            settings.min_join_coverage_pct,
            "RV_X002",
            "cross_dataset",
            f"sales.sku_id coverage to products >= {settings.min_join_coverage_pct}%",
            severity="HIGH",
        )
    )

    return results
