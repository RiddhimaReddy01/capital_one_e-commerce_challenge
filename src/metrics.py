from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import ensure_dir, write_table
from .config import Config


_SEGMENT_AGG = {
    "unique_customers": ("buyer_id", "nunique"),
    "order_count": ("order_id", "nunique"),
    "avg_profit_per_order": ("order_profit", "mean"),
    "median_profit_per_order": ("order_profit", "median"),
    "total_profit": ("order_profit", "sum"),
    "total_revenue": ("order_revenue", "sum"),
    "total_cogs": ("order_cogs", "sum"),
    "total_referral_discount": ("referral_discount_amount", "sum"),
    "total_black_friday_discount": ("black_friday_discount_amount", "sum"),
    "total_company_shipping": ("company_shipping_cost", "sum"),
}


def _add_derived_cols(df: pd.DataFrame) -> None:
    df["orders_per_customer"] = np.where(
        df["unique_customers"] > 0, df["order_count"] / df["unique_customers"], 0.0
    )
    df["total_discounts"] = df["total_referral_discount"] + df["total_black_friday_discount"]
    df["discount_pct"] = np.where(
        df["total_revenue"] > 0, (df["total_discounts"] / df["total_revenue"]) * 100, 0.0
    )
    df["cogs_pct"] = np.where(
        df["total_revenue"] > 0, (df["total_cogs"] / df["total_revenue"]) * 100, 0.0
    )
    df["shipping_per_order"] = np.where(
        df["order_count"] > 0, df["total_company_shipping"] / df["order_count"], 0.0
    )


def q1_hourly_order_volume(order_level: pd.DataFrame) -> pd.DataFrame:
    valid_years = order_level[(order_level["year"] >= 2024) & (order_level["year"] <= 2025)]
    hourly = (
        valid_years.groupby(["year", "hour_company"], as_index=False)
        .agg(order_volume=("order_id", "nunique"))
        .rename(columns={"hour_company": "hour"})
    )

    return hourly.sort_values(["year", "hour"]).reset_index(drop=True)


def q2_avg_profit_by_quarter_segment(order_level: pd.DataFrame) -> pd.DataFrame:
    valid_years = order_level[(order_level["year"] >= 2024) & (order_level["year"] <= 2025)]
    out = (
        valid_years.groupby(["quarter", "customer_segment"], as_index=False)
        .agg(
            avg_profit_per_order=("order_profit", "mean"),
            median_profit_per_order=("order_profit", "median"),
            q25_profit=("order_profit", lambda x: x.quantile(0.25)),
            q75_profit=("order_profit", lambda x: x.quantile(0.75)),
            total_profit=("order_profit", "sum"),
            order_count=("order_id", "nunique"),
            total_revenue=("order_revenue", "sum"),
        )
        .sort_values(["quarter", "customer_segment"])
        .reset_index(drop=True)
    )
    out["profit_iqr"] = out["q75_profit"] - out["q25_profit"]
    out["profit_volatility"] = out["profit_iqr"] / (out["median_profit_per_order"].abs() + 1)
    return out



def q3_referral_program_impact(order_level: pd.DataFrame) -> pd.DataFrame:
    valid_years = order_level[(order_level["year"] >= 2024) & (order_level["year"] <= 2025)]
    grouped = valid_years.assign(
        referral_group=np.where(valid_years["is_referred"] == 1, "referred", "non_referred")
    )
    by_ref = (
        grouped
        .groupby("referral_group", as_index=False)
        .agg(
            unique_customers=("buyer_id", "nunique"),
            order_count=("order_id", "nunique"),
            avg_profit_per_order=("order_profit", "mean"),
            median_profit_per_order=("order_profit", "median"),
            total_profit=("order_profit", "sum"),
            total_revenue=("order_revenue", "sum"),
            total_referral_discount=("referral_discount_amount", "sum"),
            total_black_friday_discount=("black_friday_discount_amount", "sum"),
            total_cogs=("order_cogs", "sum"),
            total_company_shipping_cost=("company_shipping_cost", "sum"),
        )
    )

    # Counterfactual: remove referral discount impact at order grain.
    grouped["counterfactual_profit"] = (
        grouped["order_revenue"] + grouped["referral_discount_amount"]
        - grouped["order_cogs"] - grouped["company_shipping_cost"]
    )
    cf = grouped.groupby("referral_group", as_index=False).agg(
        counterfactual_profit=("counterfactual_profit", "sum")
    )
    by_ref = by_ref.merge(cf, on="referral_group", how="left")
    by_ref["net_lift_from_referral"] = by_ref["counterfactual_profit"] - by_ref["total_profit"]

    # Customer quality delta: referred vs non-referred profitability gap
    avg_profit_referred = by_ref[by_ref["referral_group"] == "referred"]["avg_profit_per_order"].values
    avg_profit_non_referred = by_ref[by_ref["referral_group"] == "non_referred"]["avg_profit_per_order"].values

    if len(avg_profit_referred) > 0 and len(avg_profit_non_referred) > 0:
        profit_diff = avg_profit_referred[0] - avg_profit_non_referred[0]
        by_ref["profit_quality_delta"] = np.where(
            by_ref["referral_group"] == "referred",
            profit_diff * by_ref["order_count"],
            0.0
        )
    else:
        by_ref["profit_quality_delta"] = 0.0

    # Total referral cost: discount + customer quality loss
    by_ref["total_referral_cost"] = by_ref["net_lift_from_referral"] + by_ref["profit_quality_delta"]

    return by_ref


def q4_black_friday_impact(order_level: pd.DataFrame) -> pd.DataFrame:
    work = order_level[(order_level["year"] >= 2024) & (order_level["year"] <= 2025)].copy()
    work["order_date_company"] = pd.to_datetime(work["order_datetime_company"], errors="coerce").dt.date
    work = work[work["order_date_company"].notna()].copy()

    daily = (
        work.groupby(["year", "customer_segment", "order_date_company"], as_index=False)
        .agg(
            daily_orders=("order_id", "nunique"),
            daily_revenue=("order_revenue", "sum"),
            daily_profit=("order_profit", "sum"),
            daily_discount=("black_friday_discount_amount", "sum"),
            is_black_friday=("is_black_friday", "max"),
        )
    )

    annual = (
        daily.groupby(["year", "customer_segment"], as_index=False)
        .agg(
            annual_orders=("daily_orders", "sum"),
            annual_revenue=("daily_revenue", "sum"),
            annual_profit=("daily_profit", "sum"),
        )
    )

    bf = (
        daily[daily["is_black_friday"]]
        .groupby(["year", "customer_segment"], as_index=False)
        .agg(
            bf_orders=("daily_orders", "sum"),
            bf_revenue=("daily_revenue", "sum"),
            bf_profit=("daily_profit", "sum"),
            bf_discount_amount=("daily_discount", "sum"),
        )
    )

    baseline = (
        daily[~daily["is_black_friday"]]
        .groupby(["year", "customer_segment"], as_index=False)
        .agg(
            median_daily_orders=("daily_orders", "median"),
            median_daily_revenue=("daily_revenue", "median"),
            median_daily_profit=("daily_profit", "median"),
        )
    )

    out = annual.merge(bf, on=["year", "customer_segment"], how="left")
    out = out.merge(baseline, on=["year", "customer_segment"], how="left").fillna(0)

    out["volume_lift_x"] = np.where(
        out["median_daily_orders"] > 0, out["bf_orders"] / out["median_daily_orders"], 0
    )
    out["revenue_lift_x"] = np.where(
        out["median_daily_revenue"] > 0, out["bf_revenue"] / out["median_daily_revenue"], 0
    )

    out["bf_profit_impact"] = out["bf_profit"] - out["median_daily_profit"]
    out["bf_order_share_pct"] = np.where(out["annual_orders"] > 0, out["bf_orders"] / out["annual_orders"] * 100, 0)
    out["bf_revenue_share_pct"] = np.where(
        out["annual_revenue"] > 0, out["bf_revenue"] / out["annual_revenue"] * 100, 0
    )

    return out.sort_values(["year", "customer_segment"]).reset_index(drop=True)


def q5_customer_base_kpis(order_level: pd.DataFrame) -> pd.DataFrame:
    valid_years = order_level[(order_level["year"] >= 2024) & (order_level["year"] <= 2025)]
    seg = (
        valid_years.groupby("customer_segment", as_index=False)
        .agg(**_SEGMENT_AGG)
        .sort_values("total_profit", ascending=False)
        .reset_index(drop=True)
    )

    _add_derived_cols(seg)
    seg["profit_per_customer"] = np.where(
        seg["unique_customers"] > 0, seg["total_profit"] / seg["unique_customers"], 0.0
    )

    seg["ltv_proxy"] = seg["median_profit_per_order"] * seg["orders_per_customer"]
    median_profit_per_order = seg["median_profit_per_order"].median()
    median_order_count = seg["order_count"].median()

    high_profit = seg["median_profit_per_order"] >= median_profit_per_order
    high_volume = seg["order_count"] >= median_order_count
    seg["strategic_bucket"] = np.select(
        [high_profit & high_volume, high_profit & ~high_volume, ~high_profit & high_volume],
        ["CORE", "EXPAND", "OPTIMIZE"],
        default="DEPRIORITIZE",
    )

    return seg


def q5_quarterly_segment_analysis(order_level: pd.DataFrame) -> pd.DataFrame:
    valid_years = order_level[(order_level["year"] >= 2024) & (order_level["year"] <= 2025)]
    qtr = (
        valid_years.groupby(["quarter", "customer_segment"], as_index=False)
        .agg(**_SEGMENT_AGG)
        .sort_values(["quarter", "total_profit"], ascending=[True, False])
        .reset_index(drop=True)
    )

    _add_derived_cols(qtr)

    return qtr


def build_metrics(order_level: pd.DataFrame, settings: Config) -> dict[str, pd.DataFrame]:
    ensure_dir(settings.metrics_dir)
    outputs = {
        "q1_hourly_order_volume": q1_hourly_order_volume(order_level),
        "q2_profit_by_quarter_segment": q2_avg_profit_by_quarter_segment(order_level),
        "q3_referral_impact": q3_referral_program_impact(order_level),
        "q4_black_friday_impact": q4_black_friday_impact(order_level),
        "q5_customer_base_kpis": q5_customer_base_kpis(order_level),
        "q5_quarterly_segment_analysis": q5_quarterly_segment_analysis(order_level),
    }
    for name, df in outputs.items():
        write_table(df, settings.metrics_dir / name)
    return outputs
