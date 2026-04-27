from __future__ import annotations

import numpy as np
import pandas as pd
import pytz

from .checks import black_friday_date, normalize_id_cols
from .config import Config


def build_final_dataset_line(
    buyers_clean: pd.DataFrame,
    sales_clean: pd.DataFrame,
    products_clean: pd.DataFrame,
    settings: Config,
) -> pd.DataFrame:
    sales = sales_clean.copy()
    buyers = buyers_clean.copy()
    products = products_clean.copy()

    normalize_id_cols(sales, ["buyer_id", "sku_id"])
    normalize_id_cols(buyers, ["buyer_id"])
    normalize_id_cols(products, ["sku_id"])

    sales_buyers = sales.merge(buyers, on="buyer_id", how="inner")
    final_line = sales_buyers.merge(products, on="sku_id", how="inner", suffixes=("", "_product"))

    return final_line


def _to_company_timezone(
    order_datetimes: pd.Series,
    timezones: pd.Series,
    company_timezone: str,
) -> pd.Series:
    converted = pd.Series(pd.NaT, index=order_datetimes.index, dtype="datetime64[ns, UTC]")
    normalized_timezones = timezones.astype("string").fillna(company_timezone)

    for timezone_name, idx in normalized_timezones.groupby(normalized_timezones).groups.items():
        try:
            localized = (
                pd.to_datetime(order_datetimes.loc[idx], errors="coerce")
                .dt.tz_localize(str(timezone_name), ambiguous="NaT", nonexistent="NaT")
                .dt.tz_convert("UTC")
            )
        except (TypeError, ValueError, pytz.exceptions.UnknownTimeZoneError):
            continue
        converted.loc[idx] = localized

    return converted.dt.tz_convert(company_timezone)


def build_order_level(final_dataset_line: pd.DataFrame, settings: Config) -> pd.DataFrame:
    base = final_dataset_line.copy()
    print(f"[build_order_level] Starting with {len(base):,} line items")

    base["order_datetime"] = pd.to_datetime(base["order_datetime"], errors="coerce")
    base["price"] = pd.to_numeric(base["price"], errors="coerce")
    base["quantity"] = pd.to_numeric(base["quantity"], errors="coerce")
    base["margin"] = pd.to_numeric(base["margin"], errors="coerce")
    base["is_referred"] = pd.to_numeric(base["is_referred"], errors="coerce").fillna(0).astype(int)

    rows_before = len(base)
    base = base.dropna(subset=["order_datetime", "price", "quantity", "margin"]).copy()
    rows_after = len(base)
    rows_dropped = rows_before - rows_after
    if rows_dropped > 0:
        print(f"[build_order_level] WARNING: Dropped {rows_dropped} rows with missing order_datetime, price, quantity, or margin")

    base["timezone"] = base.get("timezone", pd.Series(index=base.index, dtype="object")).fillna(
        settings.company_timezone
    )
    base["order_datetime_company"] = _to_company_timezone(
        base["order_datetime"],
        base["timezone"],
        settings.company_timezone,
    )
    base = base[base["order_datetime_company"].notna()].copy()

    # Filter out orders with impossible dates (1969-1970 from data quality issues)
    invalid_order_mask = base["order_datetime_company"].dt.year < 2020
    invalid_order_count = invalid_order_mask.sum()
    if invalid_order_count > 0:
        print(f"[build_order_level] WARNING: Dropping {invalid_order_count} orders with order_datetime_company year < 2020")
        base = base[~invalid_order_mask].copy()

    base["line_base_revenue"] = base["price"] * base["quantity"]
    base["line_cogs"] = base["line_base_revenue"] * (1.0 - base["margin"])

    # Determine Black Friday using company timezone (Missouri)
    # This ensures all customers worldwide have consistent Black Friday definition
    years = sorted(base["order_datetime_company"].dt.year.dropna().unique())
    bf_map = {int(y): black_friday_date(int(y)) for y in years}
    base["order_date_company"] = base["order_datetime_company"].dt.date
    base["is_black_friday"] = base["order_datetime_company"].dt.year.map(bf_map).eq(base["order_date_company"])

    first_order_dt = base.groupby("buyer_id")["order_datetime_company"].min().rename("first_order_datetime_company")
    base = base.merge(first_order_dt, on="buyer_id", how="left")
    base["is_first_order"] = base["order_datetime_company"].eq(base["first_order_datetime_company"])

    base["ref_disc_rate"] = (
        ((base["is_referred"] == 1) & base["is_first_order"]).astype(float) * settings.referral_discount_rate
    )
    base["bf_disc_rate"] = base["is_black_friday"].astype(float) * settings.black_friday_discount_rate
    base["discount_rate"] = np.maximum(base["ref_disc_rate"], base["bf_disc_rate"])

    base["line_referral_discount"] = base["line_base_revenue"] * np.where(
        base["ref_disc_rate"] >= base["bf_disc_rate"], base["ref_disc_rate"], 0.0
    )
    base["line_black_friday_discount"] = base["line_base_revenue"] * np.where(
        base["bf_disc_rate"] > base["ref_disc_rate"], base["bf_disc_rate"], 0.0
    )
    base["line_net_revenue_before_shipping"] = base["line_base_revenue"] * (1.0 - base["discount_rate"])

    order_level = (
        base.groupby(["order_id", "buyer_id", "customer_segment", "state", "region"], as_index=False)
        .agg(
            order_datetime_company=("order_datetime_company", "min"),
            is_referred=("is_referred", "max"),
            is_first_order=("is_first_order", "max"),
            is_black_friday=("is_black_friday", "max"),
            order_base_revenue=("line_base_revenue", "sum"),
            order_net_revenue_before_shipping=("line_net_revenue_before_shipping", "sum"),
            order_cogs=("line_cogs", "sum"),
            referral_discount_amount=("line_referral_discount", "sum"),
            black_friday_discount_amount=("line_black_friday_discount", "sum"),
        )
    )

    order_level["customer_shipping_fee"] = np.select(
        [
            order_level["order_base_revenue"] < settings.shipping_threshold_1,
            (order_level["order_base_revenue"] >= settings.shipping_threshold_1)
            & (order_level["order_base_revenue"] < settings.shipping_threshold_2),
            order_level["order_base_revenue"] >= settings.shipping_threshold_2,
        ],
        [settings.shipping_fee_low, settings.shipping_fee_mid, settings.shipping_fee_high],
        default=settings.shipping_fee_high,
    )
    order_level["company_shipping_cost"] = settings.company_shipping_cost
    order_level["order_revenue"] = (
        order_level["order_net_revenue_before_shipping"] + order_level["customer_shipping_fee"]
    )
    order_level["order_profit"] = (
        order_level["order_revenue"] - order_level["order_cogs"] - order_level["company_shipping_cost"]
    )

    order_level["year"] = order_level["order_datetime_company"].dt.year
    order_level["quarter"] = (
        order_level["order_datetime_company"]
        .dt.tz_localize(None)
        .dt.to_period("Q")
        .astype(str)
    )
    order_level["hour_company"] = order_level["order_datetime_company"].dt.hour
    order_level["is_referral_discount"] = order_level["referral_discount_amount"] > 0

    invalid_year_mask = (order_level["year"] < 2020) | (order_level["year"] > 2026)
    invalid_year_count = invalid_year_mask.sum()
    if invalid_year_count > 0:
        print(f"[build_order_level] WARNING: Filtering {invalid_year_count} orders with invalid years (< 2020 or > 2026)")
        print(f"[build_order_level] Year distribution before filter: {order_level['year'].value_counts().sort_index().to_dict()}")
        order_level = order_level[~invalid_year_mask].copy()
        print(f"[build_order_level] Year distribution after filter: {order_level['year'].value_counts().sort_index().to_dict()}")

    return order_level
