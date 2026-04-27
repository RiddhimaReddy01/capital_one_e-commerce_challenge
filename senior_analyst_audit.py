import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("SENIOR ANALYST AUDIT: Q1-Q5 Logic & Numbers")
print("="*80)

order_level = pd.read_parquet("data/processed/final_order_level.parquet")

# ============================================================================
# Q1: HOURLY ORDER VOLUME - LOGIC CHECK
# ============================================================================
print("\n" + "="*80)
print("Q1: HOURLY ORDER VOLUME (Staffing Decision)")
print("="*80)

q1_csv = pd.read_csv("data/processed/metrics/q1_hourly_order_volume.csv")
q1_totals = q1_csv.groupby('year')['order_volume'].sum()
night_orders = q1_csv[q1_csv['hour'].between(0, 5)]['order_volume'].sum()
total_orders = q1_csv['order_volume'].sum()
night_pct = night_orders / total_orders * 100

print(f"Total orders: {total_orders:,}")
print(f"Night demand (0-5am): {night_pct:.2f}%")
print(f"Status: {'[PASS] Skeleton crew justified' if night_pct < 15 else '[WARN] Higher than expected'}")

peak_row = q1_csv.loc[q1_csv['order_volume'].idxmax()]
peak_hour = int(peak_row['hour'])
peak_vol = int(peak_row['order_volume'])
night_avg = q1_csv[q1_csv['hour'].between(0, 5)]['order_volume'].mean()
peak_ratio = peak_vol / night_avg

print(f"Peak hour: {peak_hour}:00 ({peak_vol:,} orders)")
print(f"Peak-to-night ratio: {peak_ratio:.1f}x")
print(f"Match with order-level: {'YES' if total_orders == len(order_level) else 'NO'}")

# ============================================================================
# Q2: PROFIT STABILITY - LOGIC CHECK
# ============================================================================
print("\n" + "="*80)
print("Q2: PROFIT STABILITY (Profit by Segment & Quarter)")
print("="*80)

q2_csv = pd.read_csv("data/processed/metrics/q2_profit_by_quarter_segment.csv")

# Sample verification
sample_segment = 'Tech Enthusiast'
sample_quarter = '2024Q1'
sample_data = order_level[
    (order_level['customer_segment'] == sample_segment) &
    (order_level['quarter'] == sample_quarter)
]

print(f"\nSample: {sample_segment} in {sample_quarter}")
print(f"  Order count: {sample_data['order_id'].nunique()}")
print(f"  Avg profit (calculated): ${sample_data['order_profit'].mean():.2f}")
print(f"  Median profit (calculated): ${sample_data['order_profit'].median():.2f}")

q25 = sample_data['order_profit'].quantile(0.25)
q75 = sample_data['order_profit'].quantile(0.75)
iqr = q75 - q25
volatility = iqr / (abs(sample_data['order_profit'].median()) + 1)

print(f"  IQR calculated: ${iqr:.2f}")
print(f"  Volatility calculated: {volatility:.4f}")

q2_sample = q2_csv[(q2_csv['customer_segment'] == sample_segment) & (q2_csv['quarter'] == sample_quarter)]
if len(q2_sample) > 0:
    print(f"\n  Q2 CSV median: ${q2_sample['median_profit_per_order'].values[0]:.2f}")
    print(f"  Q2 CSV volatility: {q2_sample['profit_volatility'].values[0]:.4f}")
    match = abs(sample_data['order_profit'].median() - q2_sample['median_profit_per_order'].values[0]) < 0.01
    print(f"  Match: {'YES' if match else 'NO'}")

# Volatility range
vol_mean = q2_csv['profit_volatility'].mean()
vol_min = q2_csv['profit_volatility'].min()
vol_max = q2_csv['profit_volatility'].max()
print(f"\nVolatility range: {vol_min:.2f} to {vol_max:.2f} (avg: {vol_mean:.2f})")

# ============================================================================
# Q3: REFERRAL IMPACT - LOGIC CHECK
# ============================================================================
print("\n" + "="*80)
print("Q3: REFERRAL PROGRAM IMPACT")
print("="*80)

q3_csv = pd.read_csv("data/processed/metrics/q3_referral_impact.csv")

ref_orders = order_level[order_level['is_referred'] == 1]
non_ref_orders = order_level[order_level['is_referred'] == 0]

print(f"\nReferred segment:")
print(f"  Customers: {ref_orders['buyer_id'].nunique():,}")
print(f"  Orders: {ref_orders['order_id'].nunique():,}")
print(f"  Avg profit/order: ${ref_orders['order_profit'].mean():.2f}")
print(f"  Total profit: ${ref_orders['order_profit'].sum():,.2f}")
print(f"  Referral discount: ${ref_orders['referral_discount_amount'].sum():,.2f}")

print(f"\nNon-referred segment:")
print(f"  Customers: {non_ref_orders['buyer_id'].nunique():,}")
print(f"  Orders: {non_ref_orders['order_id'].nunique():,}")
print(f"  Avg profit/order: ${non_ref_orders['order_profit'].mean():.2f}")

profit_diff = ref_orders['order_profit'].mean() - non_ref_orders['order_profit'].mean()
print(f"\nProfit difference: ${profit_diff:.2f} per order")

if profit_diff < 0:
    print(f"[CRITICAL] Referred customers are LESS profitable than non-referred!")
    print(f"  This means the referral program is subsidizing lower-value customers")
    print(f"  Discount cost: ${ref_orders['referral_discount_amount'].sum():,.2f}")
    print(f"  Customer quality issue: -${abs(profit_diff) * ref_orders['order_id'].nunique():,.0f}")
else:
    print(f"[PASS] Referred customers are more profitable")

q3_lift = q3_csv[q3_csv['referral_group'] == 'referred']['net_lift_from_referral'].values[0]
print(f"\nQ3 net lift: ${q3_lift:,.2f}")
print(f"Interpretation: Referral discount cost per order group")

# ============================================================================
# Q4: BLACK FRIDAY IMPACT - LOGIC CHECK
# ============================================================================
print("\n" + "="*80)
print("Q4: BLACK FRIDAY IMPACT")
print("="*80)

q4_csv = pd.read_csv("data/processed/metrics/q4_black_friday_impact.csv")

bf_orders = order_level[order_level['is_black_friday'] == 1]
bf_base_revenue = bf_orders['order_base_revenue'].sum()
bf_discount = bf_orders['black_friday_discount_amount'].sum()
bf_discount_rate = (bf_discount / bf_base_revenue * 100) if bf_base_revenue > 0 else 0

print(f"\nBlack Friday discount rate:")
print(f"  Base revenue: ${bf_base_revenue:,.2f}")
print(f"  Discount amount: ${bf_discount:,.2f}")
print(f"  Discount rate: {bf_discount_rate:.2f}% (policy: 20%)")
print(f"  Status: {'[PASS] Within 1%' if abs(bf_discount_rate - 20) <= 1 else '[WARN] Off by ' + str(abs(bf_discount_rate - 20))}")

total_bf_orders = bf_orders['order_id'].nunique()
non_bf_orders = order_level[order_level['is_black_friday'] == 0]
median_daily_non_bf = non_bf_orders.groupby(non_bf_orders['order_datetime_company'].dt.date)['order_id'].nunique().median()

print(f"\nBlack Friday volume lift:")
print(f"  BF orders: {total_bf_orders:,}")
print(f"  Median daily non-BF: {median_daily_non_bf:.0f}")
print(f"  Lift: {total_bf_orders / median_daily_non_bf:.2f}x")

bf_profit = bf_orders['order_profit'].sum()
non_bf_profit = non_bf_orders['order_profit'].sum()
print(f"\nBlack Friday profit impact:")
print(f"  BF profit: ${bf_profit:,.2f}")
print(f"  BF % of total profit: {bf_profit/(bf_profit+non_bf_profit)*100:.2f}%")
print(f"  BF % of total orders: {total_bf_orders/len(order_level)*100:.2f}%")
print(f"  Profit efficiency: BF contributes {bf_profit/(bf_profit+non_bf_profit)*100:.1f}% of profit from {total_bf_orders/len(order_level)*100:.1f}% of orders")

# ============================================================================
# Q5: CUSTOMER BASE KPIS - LOGIC CHECK
# ============================================================================
print("\n" + "="*80)
print("Q5: CUSTOMER BASE KPIS (Segmentation Strategy)")
print("="*80)

q5_csv = pd.read_csv("data/processed/metrics/q5_customer_base_kpis.csv")

print("\nSegment Rankings (by profit):")
for idx, row in q5_csv.sort_values('total_profit', ascending=False).iterrows():
    print(f"  {row['customer_segment']:20} | Profit: ${row['total_profit']:>12,.0f} | LTV: ${row['ltv_proxy']:>8.0f} | Bucket: {row['strategic_bucket']}")

# LTV proxy check
sample_seg = q5_csv.iloc[0]['customer_segment']
sample_seg_data = order_level[order_level['customer_segment'] == sample_seg]
sample_ltv = sample_seg_data['order_profit'].median() * (sample_seg_data['order_id'].nunique() / sample_seg_data['buyer_id'].nunique())
q5_ltv = q5_csv[q5_csv['customer_segment'] == sample_seg]['ltv_proxy'].values[0]

print(f"\nLTV Proxy verification ({sample_seg}):")
print(f"  Calculated: ${sample_ltv:.2f}")
print(f"  Q5 CSV: ${q5_ltv:.2f}")
print(f"  Match: {'YES' if abs(sample_ltv - q5_ltv) < 0.1 else 'NO'}")

# Strategic bucket validation
median_profit = q5_csv['median_profit_per_order'].median()
median_volume = q5_csv['order_count'].median()
print(f"\nStrategic bucket thresholds:")
print(f"  Median profit/order: ${median_profit:.2f}")
print(f"  Median order volume: {median_volume:,.0f}")

bucket_errors = []
for idx, row in q5_csv.iterrows():
    high_profit = row['median_profit_per_order'] >= median_profit
    high_volume = row['order_count'] >= median_volume

    if high_profit and high_volume:
        expected = 'CORE'
    elif high_profit and not high_volume:
        expected = 'EXPAND'
    elif not high_profit and high_volume:
        expected = 'OPTIMIZE'
    else:
        expected = 'DEPRIORITIZE'

    if expected != row['strategic_bucket']:
        bucket_errors.append(f"  {row['customer_segment']}: expected {expected}, got {row['strategic_bucket']}")

if bucket_errors:
    print("[WARN] Bucket assignment errors:")
    for err in bucket_errors:
        print(err)
else:
    print("[PASS] All bucket assignments correct")

print("\n" + "="*80)
print("AUDIT COMPLETE")
print("="*80)
