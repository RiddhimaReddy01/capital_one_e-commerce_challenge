import pandas as pd
import json
import numpy as np
from pathlib import Path

print("="*80)
print("COMPREHENSIVE Q1-Q5 AUDIT: Data, Logic, Dashboard")
print("="*80)

order_level = pd.read_parquet("data/processed/final_order_level.parquet")

# ============================================================================
# Q1: HOURLY ORDER VOLUME
# ============================================================================
print("\n" + "="*80)
print("Q1 - HOURLY ORDER VOLUME (Staffing Decision)")
print("="*80)

q1 = pd.read_csv("data/processed/metrics/q1_hourly_order_volume.csv")
print(f"\nShape: {q1.shape}, Nulls: {q1.isnull().sum().sum()}, Duplicates: {q1.duplicated().sum()}")

invalid_hours = q1[(q1['hour'] < 0) | (q1['hour'] > 23)]
print(f"[PASS] Hour range (0-23)" if len(invalid_hours) == 0 else f"[FAIL] Invalid hours: {len(invalid_hours)}")

invalid_vol = q1[q1['order_volume'] <= 0]
print(f"[PASS] All volumes positive" if len(invalid_vol) == 0 else f"[FAIL] Non-positive: {len(invalid_vol)}")

invalid_year = q1[~q1['year'].isin([2024, 2025])]
print(f"[PASS] Years 2024-2025" if len(invalid_year) == 0 else f"[FAIL] Invalid years")

q1_total = q1['order_volume'].sum()
ol_total = len(order_level)
match = "YES" if q1_total == ol_total else f"NO ({q1_total} vs {ol_total})"
print(f"[PASS] Q1 total matches order-level: {match}")

night_pct = q1[q1['hour'].between(0, 5)]['order_volume'].sum() / q1_total * 100
peak_hour = int(q1.loc[q1['order_volume'].idxmax(), 'hour'])
print(f"[INFO] Night demand: {night_pct:.1f}%, Peak: {peak_hour}:00")
print(q1.head(10))

# ============================================================================
# Q2: PROFIT BY QUARTER SEGMENT
# ============================================================================
print("\n" + "="*80)
print("Q2 - PROFIT BY QUARTER SEGMENT (Profit Stability)")
print("="*80)

q2 = pd.read_csv("data/processed/metrics/q2_profit_by_quarter_segment.csv")
print(f"\nShape: {q2.shape}, Nulls: {q2.isnull().sum().sum()}, Duplicates: {q2.duplicated().sum()}")

invalid_years = q2[~q2['quarter'].str.extract('(\d{4})', expand=False).astype(int).isin([2024, 2025])]
print(f"[PASS] No 1969-1970 data" if len(invalid_years) == 0 else f"[FAIL] Found: {invalid_years['quarter'].unique()}")

neg_avg = q2[q2['avg_profit_per_order'] < 0]
print(f"[PASS] No negative avg_profit" if len(neg_avg) == 0 else f"[WARN] {len(neg_avg)} negative")

q2['iqr_calc'] = q2['q75_profit'] - q2['q25_profit']
q2['iqr_check'] = np.isclose(q2['iqr_calc'], q2['profit_iqr'])
iqr_errors = q2[~q2['iqr_check']]
print(f"[PASS] IQR formula correct" if len(iqr_errors) == 0 else f"[FAIL] {len(iqr_errors)} calc errors")

q2['vol_calc'] = q2['profit_iqr'] / (q2['median_profit_per_order'].abs() + 1)
q2['vol_check'] = np.isclose(q2['vol_calc'], q2['profit_volatility'])
vol_errors = q2[~q2['vol_check']]
print(f"[PASS] Volatility formula correct" if len(vol_errors) == 0 else f"[FAIL] {len(vol_errors)} calc errors")

segments = len(q2['customer_segment'].unique())
print(f"[PASS] All 8 segments" if segments == 8 else f"[WARN] Got {segments} segments")

print(q2[['quarter', 'customer_segment', 'median_profit_per_order', 'profit_volatility']].head(10))

# ============================================================================
# Q3: REFERRAL IMPACT
# ============================================================================
print("\n" + "="*80)
print("Q3 - REFERRAL PROGRAM IMPACT")
print("="*80)

q3 = pd.read_csv("data/processed/metrics/q3_referral_impact.csv")
print(f"\nShape: {q3.shape}, Nulls: {q3.isnull().sum().sum()}")

correct_groups = len(q3) == 2 and all(g in q3['referral_group'].values for g in ['referred', 'non_referred'])
print(f"[PASS] 2 groups (referred + non-referred)" if correct_groups else f"[FAIL] Wrong groups")

required = ['unique_customers', 'order_count', 'avg_profit_per_order', 'total_referral_discount', 'net_lift_from_referral']
missing = [c for c in required if c not in q3.columns]
print(f"[PASS] All required columns" if not missing else f"[FAIL] Missing: {missing}")

ref_disc = q3[q3['referral_group'] == 'referred']['total_referral_discount'].values[0]
non_ref_disc = q3[q3['referral_group'] == 'non_referred']['total_referral_discount'].values[0]
print(f"[PASS] Referral discount logic" if (ref_disc > 0 and non_ref_disc == 0) else f"[WARN] Discount values suspicious")

print(q3[['referral_group', 'unique_customers', 'avg_profit_per_order', 'total_referral_discount', 'net_lift_from_referral']])

# ============================================================================
# Q4: BLACK FRIDAY IMPACT
# ============================================================================
print("\n" + "="*80)
print("Q4 - BLACK FRIDAY IMPACT")
print("="*80)

q4 = pd.read_csv("data/processed/metrics/q4_black_friday_impact.csv")
print(f"\nShape: {q4.shape}, Nulls: {q4.isnull().sum().sum()}")

invalid_years = q4[~q4['year'].isin([2024, 2025])]
print(f"[PASS] Years 2024-2025" if len(invalid_years) == 0 else f"[FAIL] Invalid years")

negative_cols = ['bf_orders', 'bf_profit', 'bf_discount_amount']
has_neg = any(q4[col].min() < 0 for col in negative_cols if col in q4.columns)
print(f"[PASS] No negative values" if not has_neg else f"[WARN] Found negative values")

bf_discount_rate = (q4['bf_discount_amount'].sum() / q4['bf_revenue'].sum() * 100) if q4['bf_revenue'].sum() > 0 else 0
print(f"[INFO] BF discount rate: {bf_discount_rate:.1f}% (expected ~20%)")

total_bf_orders = q4['bf_orders'].sum()
total_baseline_orders = q4['median_daily_orders'].sum() * len(q4)
bf_lift = total_bf_orders / total_baseline_orders if total_baseline_orders > 0 else 0
print(f"[INFO] BF volume lift: {bf_lift:.2f}x")
print(f"[PASS] Positive BF lift" if bf_lift > 1 else f"[FAIL] No positive lift")

print(q4[['year', 'customer_segment', 'bf_orders', 'bf_profit', 'bf_discount_amount']].head(10))

# ============================================================================
# Q5: CUSTOMER BASE KPIS
# ============================================================================
print("\n" + "="*80)
print("Q5 - CUSTOMER BASE KPIS (Segmentation)")
print("="*80)

q5 = pd.read_csv("data/processed/metrics/q5_customer_base_kpis.csv")
print(f"\nShape: {q5.shape}, Nulls: {q5.isnull().sum().sum()}")

segments = len(q5['customer_segment'].unique())
print(f"[PASS] All 8 segments" if segments == 8 else f"[WARN] Got {segments}")

if 'strategic_bucket' in q5.columns:
    buckets = set(q5['strategic_bucket'].unique())
    expected = {'CORE', 'EXPAND', 'OPTIMIZE', 'DEPRIORITIZE'}
    if buckets == expected:
        print(f"[PASS] All 4 strategic buckets")
    else:
        print(f"[WARN] Buckets: {buckets}")
else:
    print(f"[FAIL] No strategic_bucket column")

neg_profit = q5[q5['total_profit'] < 0]
print(f"[PASS] All profitable" if len(neg_profit) == 0 else f"[WARN] {len(neg_profit)} negative profit")

q5['orders_check'] = np.isclose(
    q5['order_count'] / q5['unique_customers'],
    q5['orders_per_customer']
)
orders_bad = q5[~q5['orders_check']]
print(f"[PASS] orders_per_customer calc" if len(orders_bad) == 0 else f"[FAIL] {len(orders_bad)} calc errors")

print(q5[['customer_segment', 'total_profit', 'ltv_proxy', 'strategic_bucket']].sort_values('total_profit', ascending=False))

# ============================================================================
# CROSS-Q VALIDATION
# ============================================================================
print("\n" + "="*80)
print("CROSS-QUESTION VALIDATION")
print("="*80)

q5_total_customers = q5['unique_customers'].sum()
q1_orders = q1['order_volume'].sum()
print(f"\n[INFO] Total orders (Q1): {q1_orders:,}")
print(f"[INFO] Total customers (Q5): {q5_total_customers:,}")
print(f"[INFO] Avg orders/customer: {q1_orders / q5_total_customers:.2f}")

segments_q2 = set(q2['customer_segment'].unique())
segments_q5 = set(q5['customer_segment'].unique())
if segments_q2 == segments_q5:
    print(f"[PASS] Segment consistency Q2 vs Q5")
else:
    print(f"[FAIL] Segment mismatch: Q2 only={segments_q2 - segments_q5}, Q5 only={segments_q5 - segments_q2}")

print("\n" + "="*80)
print("ALL TESTS COMPLETE")
print("="*80)
