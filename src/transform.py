from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .checks import normalize_text_col
from .config import Config


def _mode_or_nan(x: pd.Series) -> object:
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan


def clean_sales(settings: Config) -> pd.DataFrame:
    raw_file = settings.raw_data_path / "sales.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw sales.csv at {raw_file}")

    print(f"[clean_sales] Loading raw sales data from {raw_file}")
    sales = pd.read_csv(raw_file)
    print(f"[clean_sales] Loaded {len(sales):,} raw sales records")
    for col in ["order_id", "buyer_id", "sku_id"]:
        sales[col] = sales[col].astype("string").str.strip()

    sales["order_status"] = sales["order_status"].replace("Deliverred", "Delivered")
    sales = sales.dropna(subset=["order_status"]).copy()
    sales = sales.dropna(subset=["order_id", "buyer_id", "sku_id"]).copy()

    sales["order_datetime_raw"] = sales["order_datetime"].astype("string")
    strict_dt = pd.to_datetime(sales["order_datetime_raw"], format="%m/%d/%y %H:%M", errors="coerce")
    unresolved_mask = strict_dt.isna()
    fallback_raw = sales.loc[unresolved_mask, "order_datetime_raw"]
    fallback_dt = pd.Series(pd.NaT, index=fallback_raw.index, dtype="datetime64[ns]")
    for fmt in ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        remaining = fallback_dt.isna()
        if not remaining.any():
            break
        fallback_dt.loc[remaining] = pd.to_datetime(fallback_raw.loc[remaining], format=fmt, errors="coerce")
    strict_dt.loc[unresolved_mask] = fallback_dt
    sales["order_datetime"] = strict_dt

    strict_parsed_count = int((~unresolved_mask).sum())
    fallback_parsed_count = int(fallback_dt.notna().sum())
    datetime_unparsed_dropped_count = int(sales["order_datetime"].isna().sum())

    sales = sales[sales["order_datetime"].notna()].copy()
    min_date = sales["order_datetime"].min()
    max_date = sales["order_datetime"].max()
    print(f"[clean_sales] Date range: {min_date} to {max_date}")

    sales = sales[
        (sales["order_datetime"] >= min_date)
        & (sales["order_datetime"] <= max_date)
    ].copy()
    sales = sales.drop_duplicates(subset=["order_id", "buyer_id", "sku_id"], keep="first").copy()

    # Robust quantity handling from sales EDA: keep valid positive quantities and cap extreme spikes.
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce")
    invalid_qty_mask = sales["quantity"].isna() | (sales["quantity"] <= 0)
    invalid_quantity_dropped_count = int(invalid_qty_mask.sum())
    if invalid_quantity_dropped_count > 0:
        print(f"[clean_sales] WARNING: Dropping {invalid_quantity_dropped_count} rows with invalid quantity (null or <= 0)")
    sales = sales[~invalid_qty_mask].copy()
    qty_cap = float(np.floor(sales["quantity"].quantile(0.999))) if len(sales) else 3.0
    qty_cap = max(3.0, qty_cap)
    print(f"[clean_sales] Quantity cap set to: {qty_cap}")
    sales["quantity_original"] = sales["quantity"]
    sales["quantity_was_capped"] = sales["quantity"] > qty_cap
    sales["quantity_cap_value"] = qty_cap
    quantity_capped_row_count = int(sales["quantity_was_capped"].sum())
    if quantity_capped_row_count > 0:
        print(f"[clean_sales] Capping {quantity_capped_row_count} rows with quantity > {qty_cap}")
    sales["quantity"] = sales["quantity"].clip(upper=qty_cap)

    sales.attrs["cleaning_audit"] = {
        "datetime_strict_parsed_count": strict_parsed_count,
        "datetime_fallback_parsed_count": fallback_parsed_count,
        "datetime_unparsed_dropped_count": datetime_unparsed_dropped_count,
        "invalid_quantity_dropped_count": invalid_quantity_dropped_count,
        "quantity_cap_value": qty_cap,
        "quantity_capped_row_count": quantity_capped_row_count,
    }

    sales["sku_id"] = sales["sku_id"].astype("string").str.strip().str.upper()
    sales = sales.drop(columns=["order_datetime_raw"]).copy()
    return sales


def clean_products(settings: Config) -> pd.DataFrame:
    vendor_dir = settings.raw_data_path / "Vendor Datasets"
    if not vendor_dir.exists():
        raise FileNotFoundError(f"Missing Vendor Datasets directory at {vendor_dir}")

    files = glob.glob(str(vendor_dir / "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {vendor_dir}")

    print(f"[clean_products] Loading {len(files)} product files from {vendor_dir}")
    products = pd.concat([pd.read_csv(p) for p in sorted(files)], ignore_index=True)
    print(f"[clean_products] Loaded {len(products):,} total product records")

    for col in ["subcategory_id", "vendor", "product_num"]:
        if col in products.columns:
            products[col] = normalize_text_col(products[col])
    for col in ["category_name", "subcategory_name", "vendor"]:
        if col in products.columns:
            products[col] = normalize_text_col(products[col])

    if "price" in products.columns:
        products["price"] = pd.to_numeric(
            products["price"].astype(str).str.replace("$", "", regex=False), errors="coerce"
        )

    if "pct_profit_margin" in products.columns:
        products["pct_profit_margin"] = pd.to_numeric(products["pct_profit_margin"], errors="coerce")
    if "profit_margin" in products.columns:
        products["profit_margin"] = pd.to_numeric(products["profit_margin"], errors="coerce")

    # Validate margin column exists in some form
    if "profit_margin" not in products.columns and "pct_profit_margin" not in products.columns:
        raise ValueError("ERROR: Neither profit_margin nor pct_profit_margin found in products data")

    products["margin"] = products.get("profit_margin")
    if "pct_profit_margin" in products.columns:
        products["margin"] = products["margin"].fillna(products["pct_profit_margin"] / 100.0)

    products["margin_imputed"] = products["margin"].isna()
    if "subcategory_id" in products.columns:
        subcat_avg = products.groupby("subcategory_id")["margin"].transform("mean")
        products["margin"] = products["margin"].fillna(subcat_avg)
    products["margin_imputed"] = products["margin_imputed"] & products["margin"].notna()

    products["active_product_num"] = pd.to_numeric(products.get("active_product"), errors="coerce")
    products["currently_active_vendor_num"] = pd.to_numeric(
        products.get("currently_active_vendor"), errors="coerce"
    )
    products["current_stock_num"] = pd.to_numeric(products.get("current_stock"), errors="coerce")
    products["vendor_product_conflict"] = (
        (products["active_product_num"] == 1) & (products["currently_active_vendor_num"] == 0)
    ).astype("Int64")
    products["inactive_with_stock_flag"] = (
        (products["active_product_num"] == 0) & (products["current_stock_num"] > 0)
    )

    products["sku_id"] = (
        products["subcategory_id"].astype("string")
        + "-"
        + products["vendor"].astype("string")
        + "-"
        + products["product_num"].astype("string")
    )
    products["sku_id"] = products["sku_id"].astype("string").str.strip().str.upper()

    return products


def clean_buyers(settings: Config, sales_clean: pd.DataFrame) -> pd.DataFrame:
    raw_file = settings.raw_data_path / "buyer.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw buyer.csv at {raw_file}")

    print(f"[clean_buyers] Loading buyer data from {raw_file}")
    buyers = pd.read_csv(raw_file)
    print(f"[clean_buyers] Loaded {len(buyers):,} buyer records")
    buyers["buyer_id"] = buyers["buyer_id"].astype("string").str.strip()
    buyers["signup_date"] = pd.to_datetime(
        buyers["signup_date"], format="%m/%d/%y %H:%M", errors="coerce"
    )

    if "is_active_buyer" in buyers.columns:
        buyers = buyers.drop(columns=["is_active_buyer"]).copy()

    state_tz = buyers.dropna(subset=["timezone"]).groupby("state")["timezone"].agg(_mode_or_nan)
    tz_mask = buyers["timezone"].isna() & buyers["state"].notna()
    buyers.loc[tz_mask, "timezone"] = buyers.loc[tz_mask, "state"].map(state_tz)

    state_region = buyers.dropna(subset=["region"]).groupby("state")["region"].agg(_mode_or_nan)
    region_mask = buyers["region"].isna() & buyers["state"].notna()
    buyers.loc[region_mask, "region"] = buyers.loc[region_mask, "state"].map(state_region)

    buyers = buyers[buyers["buyer_id"].notna()].copy()
    if settings.buyers_coverage_filter:
        sales_ids = set(sales_clean["buyer_id"].dropna().unique())
        buyers = buyers[buyers["buyer_id"].isin(sales_ids)].copy()

    # EDA-to-prod migration: infer missing signup_date from first observed order timestamp
    first_order = (
        sales_clean.sort_values(["buyer_id", "order_datetime"])
        .groupby("buyer_id", as_index=False)["order_datetime"]
        .min()
        .rename(columns={"order_datetime": "first_order_datetime"})
    )
    buyers = buyers.merge(first_order, on="buyer_id", how="left")
    signup_fill_mask = buyers["signup_date"].isna() & buyers["first_order_datetime"].notna()
    buyers.loc[signup_fill_mask, "signup_date"] = buyers.loc[signup_fill_mask, "first_order_datetime"]
    buyers = buyers.drop(columns=["first_order_datetime"]).copy()

    buyers["completeness_score"] = (
        buyers["signup_date"].notna().astype(int)
        + buyers["region"].notna().astype(int)
        + buyers["timezone"].notna().astype(int)
        + buyers["state"].notna().astype(int)
    )
    buyers = (
        buyers.sort_values(["buyer_id", "completeness_score"], ascending=[True, False])
        .drop_duplicates(subset=["buyer_id"], keep="first")
        .drop(columns=["completeness_score"])
        .copy()
    )

    buyers["is_referred_invalid"] = ~buyers["is_referred"].isin([0, 1])
    buyers.loc[buyers["is_referred_invalid"], "is_referred"] = pd.NA
    buyers["is_referred"] = buyers["is_referred"].astype("Int64")

    return buyers


def build_clean_layer(settings: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = clean_sales(settings)
    products = clean_products(settings)
    buyers = clean_buyers(settings, sales)
    return buyers, sales, products
