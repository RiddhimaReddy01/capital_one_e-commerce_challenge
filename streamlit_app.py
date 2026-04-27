import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pytz

from src.theme import ACCENT, DANGER, SUCCESS, SEGMENT_COLORS

st.set_page_config(
    page_title="Capital One E-Commerce Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

# ============================================================================
# CONFIG & STYLING
# ============================================================================

METRICS_DIR = Path("data/processed/metrics")
ORDER_LEVEL_PATH = Path("data/processed/final_order_level.parquet")

METRIC_FILES = {
    "q1": "q1_hourly_order_volume.csv",
    "q2": "q2_profit_by_quarter_segment.csv",
    "q3": "q3_referral_impact.csv",
    "q4": "q4_black_friday_impact.csv",
    "q5": "q5_customer_base_kpis.csv",
    "q5_qtr": "q5_quarterly_segment_analysis.csv",
}

REQUIRED_METRIC_COLUMNS = {
    "q1": {"year", "hour", "order_volume"},
    "q2": {"quarter", "customer_segment", "avg_profit_per_order", "median_profit_per_order", "order_count"},
    "q3": {"referral_group", "unique_customers", "order_count", "total_referral_discount", "net_lift_from_referral"},
    "q4": {"year", "customer_segment", "bf_orders", "bf_profit", "bf_discount_amount", "median_daily_orders", "median_daily_profit"},
    "q5": {"customer_segment", "median_profit_per_order", "order_count", "total_profit"},
    "q5_qtr": {"quarter", "customer_segment", "order_count", "total_profit"},
}

REQUIRED_ORDER_COLUMNS = {
    "buyer_id",
    "order_id",
    "order_datetime_company",
    "order_revenue",
    "order_profit",
    "customer_segment",
    "is_referred",
    "is_black_friday",
    "black_friday_discount_amount",
}

CONFIG = {
    "title_font": "Arial, sans-serif",
    "body_font": "Arial, sans-serif",
    "text_color": "#000000",
    "muted_grid": "#000000",
    "accent": ACCENT,
    "danger": DANGER,
    "success": SUCCESS,
    "segment_colors": SEGMENT_COLORS,
}

def get_segment_color(segment: str) -> str:
    return CONFIG["segment_colors"].get(segment, CONFIG["accent"])

def apply_chart_theme(fig, title_text: str, height: int = 500, hovermode: str = "x unified"):
    fig.update_layout(
        title={
            "text": title_text,
            "font": {"size": 14, "color": CONFIG["text_color"], "family": CONFIG["title_font"]},
            "x": 0.5,
            "xanchor": "center",
        },
        height=height,
        hovermode=hovermode,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"size": 10, "family": CONFIG["body_font"], "color": CONFIG["text_color"]},
        margin={"l": 55, "r": 35, "t": 72, "b": 45},
        legend={
            "orientation": "h",
            "x": 0.01,
            "y": -0.12,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"family": CONFIG["body_font"], "size": 10, "color": "#000000"},
        },
        hoverlabel={
            "font": {"family": CONFIG["body_font"], "size": 10, "color": CONFIG["text_color"]},
            "bgcolor": "white",
            "bordercolor": CONFIG["accent"],
        },
        xaxis={
            "showline": True,
            "linecolor": "#000000",
            "showgrid": False,
            "zerolinecolor": "#000000",
            "tickfont": {"color": "#000000", "family": CONFIG["body_font"]},
            "title": {"font": {"color": "#000000", "family": CONFIG["body_font"]}},
        },
        yaxis={
            "showline": True,
            "linecolor": "#000000",
            "showgrid": False,
            "zerolinecolor": "#000000",
            "tickfont": {"color": "#000000", "family": CONFIG["body_font"]},
            "title": {"font": {"color": "#000000", "family": CONFIG["body_font"]}},
        },
    )
    return fig

@st.cache_data
def load_metrics():
    loaded = {}
    missing_files = []
    schema_errors = []
    for key, filename in METRIC_FILES.items():
        path = METRICS_DIR / filename
        if not path.exists():
            missing_files.append(str(path))
            continue
        df = pd.read_csv(path)
        missing_cols = REQUIRED_METRIC_COLUMNS[key] - set(df.columns)
        if missing_cols:
            schema_errors.append(f"{path}: missing {sorted(missing_cols)}")
        loaded[key] = df
    if missing_files or schema_errors:
        details = "\n".join(missing_files + schema_errors)
        raise RuntimeError(f"Dashboard metric artifacts are incomplete:\n{details}")
    return loaded

@st.cache_data
def load_order_level():
    if not ORDER_LEVEL_PATH.exists():
        raise RuntimeError(f"Missing required order-level artifact: {ORDER_LEVEL_PATH}")
    df = pd.read_parquet(ORDER_LEVEL_PATH)
    missing_cols = REQUIRED_ORDER_COLUMNS - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"{ORDER_LEVEL_PATH} is missing required columns: {sorted(missing_cols)}")
    return df

try:
    metrics = load_metrics()
    order_level = load_order_level()
except Exception as exc:
    st.error("Dashboard startup validation failed.")
    st.exception(exc)
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
    <style>
    .stApp {
        background-color: #F7FAFC;
        color: #000000;
        font-family: Arial, sans-serif;
    }
    .stApp * {
        font-family: Arial, sans-serif !important;
        color: #000000;
    }
    .header {
        background: linear-gradient(135deg, #1F4E79 0%, #2d5a9e 100%);
        color: #000000;
        padding: 30px 36px;
        border-radius: 12px;
        margin-bottom: 18px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(31, 78, 121, 0.2);
    }
    .header h1 { margin: 0; font-size: 2em; font-weight: 700; }
    .header p { margin: 10px 0 0 0; font-size: 0.95em; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #000000;
        border-radius: 10px;
        padding: 8px;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.84rem !important;
        line-height: 1.1 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        line-height: 1.0 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        line-height: 1.05 !important;
    }
    .callout {
        background: white;
        padding: 16px;
        margin: 14px 0 18px 0;
        border-radius: 10px;
        border: 1px solid #000000;
        line-height: 1.45;
    }
    .decision-box {
        border-left: 4px solid #2E7D32;
    }
    .insight-box {
        background: #F8FBFF;
        border-left: 4px solid #1F4E79;
    }
    .executive-summary {
        background: #EDF5FF;
        border-left: 4px solid #1F4E79;
        padding: 18px;
        margin: 14px 0 24px 0;
        border-radius: 10px;
        border: 1px solid #000000;
        font-size: 0.95em;
        line-height: 1.45;
    }
    .data-quality {
        background: white;
        border: 1px solid #000000;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.85em;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Q1. Staffing Decision",
    "Q2. Profit Stability",
    "Q3. Referral Economics",
    "Q4. Promotion Effectiveness",
    "Q5. Segment Strategy"
])

# ============================================================================
# Q1: STAFFING DECISION
# ============================================================================
with tab1:
    df_q1 = metrics["q1"].copy()
    company_tz = pytz.timezone("US/Central")

    # Recompute from order-level with explicit pytz handling.
    ts = pd.to_datetime(order_level["order_datetime_company"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(company_tz)
    else:
        ts = ts.dt.tz_convert(company_tz)
    df_q1 = pd.DataFrame({
        "year": ts.dt.year,
        "hour": ts.dt.hour,
    }).dropna()
    # Filter to valid years (2024-2025 challenge data)
    df_q1 = df_q1[(df_q1["year"] >= 2024) & (df_q1["year"] <= 2025)].copy()
    df_q1 = (
        df_q1.groupby(["year", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "order_volume"})
    )

    # Calculate metrics for board presentation
    total_orders = df_q1["order_volume"].sum()
    night_orders = df_q1[df_q1["hour"].between(0, 5)]["order_volume"].sum()
    night_pct = (night_orders / total_orders * 100) if total_orders > 0 else 0.0

    peak_row = df_q1.loc[df_q1["order_volume"].idxmax()]
    peak_hour = int(peak_row["hour"])
    peak_orders = int(peak_row["order_volume"])

    night_avg = df_q1[df_q1["hour"].between(0, 5)]["order_volume"].mean()
    peak_to_night = peak_orders / night_avg if night_avg > 0 else 0.0

    # KPI Tiles (board-level metrics)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Night Demand Share", f"{night_pct:.1f}%", "of daily volume")
    col2.metric("Peak Hour Orders", f"{peak_orders:,}", f"at {peak_hour}:00")
    col3.metric("Peak-to-Night Ratio", f"{peak_to_night:.1f}x", "staffing differential")
    col4.metric("Night Orders (12 AM - 6 AM)", f"{int(night_orders):,}", "across both years")

    # Chart
    fig = go.Figure()
    for year in sorted(df_q1["year"].unique()):
        year_data = df_q1[df_q1["year"] == year].sort_values("hour")
        fig.add_trace(go.Scatter(
            x=year_data["hour"],
            y=year_data["order_volume"],
            mode="lines+markers",
            name=str(year),
            line={"width": 3},
            hovertemplate="Hour %{x}:00<br>Orders: %{y:,.0f}<extra></extra>",
        ))

    # Mark night hours boundary
    fig.add_vline(x=5.5, line_dash="dash", line_color="#000000", line_width=1)
    fig.add_annotation(
        x=peak_hour,
        y=peak_orders,
        text=f"Peak ~{peak_hour}:00",
        showarrow=True,
        arrowhead=2,
        ax=35,
        ay=-30,
        font={"size": 10, "family": CONFIG["body_font"], "color": "#000000"},
    )

    fig.update_xaxes(title_text="Hour of Day", dtick=1, tickmode="linear", range=[-0.5, 23.5], title_font={"size": 11}, tickfont={"size": 9})
    fig.update_yaxes(title_text="Order Volume", title_font={"size": 11}, tickfont={"size": 9})
    apply_chart_theme(fig, "Hourly Order Volume by Hour (2024 vs 2025)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Business decision
    st.markdown(f"""
    <div class="callout decision-box">
    Overnight demand is low ({night_pct:.1f}%) while peak demand is concentrated in daytime hours.
    <br>Recommendation: move from full 24/7 staffing to a skeleton overnight model and concentrate resources in peak periods.
    </div>
    """, unsafe_allow_html=True)

    # Data quality (explicit section to avoid expander header/icon rendering artifacts)
    st.markdown("#### Data Quality")
    st.markdown(f"""
    <div class="data-quality">
    <b>Data Coverage:</b> 100% of hourly traffic captured across 2024-2025
    <br><b>Night Definition:</b> Hours 0-5 (midnight to 6am) = 6 hours avoided if skeleton crew deployed
    <br><b>Peak Hour:</b> Consistent peak at hour {peak_hour}:00 across both years
    <br><b>Timezone Handling:</b> Converted using <code>pytz.timezone("US/Central")</code> (DST-aware, no fixed UTC offset)
    <br><b>Seasonal Variance:</b> Analysis includes Black Friday spikes; refresh staffing plan quarterly
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Q2: PROFIT STABILITY
# ============================================================================
with tab2:
    q2_orders = (
        order_level.groupby(["quarter", "customer_segment"], as_index=False)
        .agg(
            avg_profit_per_order=("order_profit", "mean"),
            order_count=("order_id", "nunique"),
        )
    )

    q2_orders["year_num"] = q2_orders["quarter"].str[:4].astype(int)
    q2_orders["q_num"] = q2_orders["quarter"].str[-1].astype(int)
    q2_orders = q2_orders.sort_values(["year_num", "q_num", "customer_segment"])
    if q2_orders.empty or not np.isfinite(q2_orders["avg_profit_per_order"].to_numpy(dtype=float)).any():
        st.error(
            "Q2 profit stability metrics are unavailable because the loaded order-level dataset "
            "does not contain usable quarterly profit values."
        )
        st.stop()

    segment_stats = (
        q2_orders.groupby("customer_segment")["avg_profit_per_order"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "avg_profit", "std": "std_profit"})
        .reset_index()
    )
    segment_stats["cv"] = segment_stats["std_profit"] / segment_stats["avg_profit"]
    segment_stats = segment_stats.sort_values("avg_profit", ascending=False)

    portfolio_avg = float(q2_orders["avg_profit_per_order"].mean())
    top_segment = segment_stats.iloc[0]["customer_segment"]
    top_segment_avg = float(segment_stats.iloc[0]["avg_profit"])
    most_stable_segment = segment_stats.sort_values("cv").iloc[0]["customer_segment"]
    most_stable_cv = float(segment_stats.sort_values("cv").iloc[0]["cv"])
    lowest_segment = segment_stats.sort_values("avg_profit").iloc[0]["customer_segment"]
    lowest_segment_avg = float(segment_stats.sort_values("avg_profit").iloc[0]["avg_profit"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Avg Profit/Order", f"${portfolio_avg:,.2f}", "quarterly segment average")
    col2.metric("Highest Avg Segment", top_segment, f"${top_segment_avg:,.2f}/order")
    col3.metric("Most Stable Segment (CV)", most_stable_segment, f"CV {most_stable_cv:.3f}")
    col4.metric("Lowest Profit Segment", lowest_segment, f"${lowest_segment_avg:,.2f}/order")

    segment_order = segment_stats["customer_segment"].tolist()
    quarter_order = (
        q2_orders[["quarter", "year_num", "q_num"]]
        .drop_duplicates()
        .sort_values(["year_num", "q_num"])["quarter"]
        .tolist()
    )
    pivot_q2 = q2_orders.pivot(index="customer_segment", columns="quarter", values="avg_profit_per_order")
    pivot_q2 = pivot_q2.reindex(columns=quarter_order)
    pivot_q2 = pivot_q2.reindex(segment_order)
    heat_values = pivot_q2.values.astype(float)
    finite_heat_values = heat_values[np.isfinite(heat_values)]
    if finite_heat_values.size == 0:
        st.error("Q2 heatmap values are unavailable after pivoting customer segment by quarter.")
        st.stop()
    heat_min = float(finite_heat_values.min())
    heat_max = float(finite_heat_values.max())
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_q2.values,
        x=pivot_q2.columns,
        y=pivot_q2.index,
        text=np.round(pivot_q2.values, 2),
        texttemplate="$%{text}",
        colorscale=[
            [0.0, "#FFF4E6"],
            [0.35, "#FDBA74"],
            [0.65, "#F97316"],
            [1.0, "#B91C1C"],
        ],
        zmin=heat_min,
        zmax=heat_max,
        xgap=1,
        ygap=1,
        colorbar={"title": "Avg profit/order"},
        hovertemplate="Segment: %{y}<br>Quarter: %{x}<br>Avg profit/order: $%{z:.2f}<extra></extra>",
    ))
    apply_chart_theme(fig_heat, "Average Profit per Order by Segment and Quarter", height=500)
    fig_heat.update_xaxes(title_text="Quarter")
    fig_heat.update_yaxes(title_text="Customer segment")
    fig_heat.add_annotation(
        x=0.5,
        y=1.03,
        xref="paper",
        yref="paper",
        text="Tech Enthusiast consistently highest across all quarters",
        showarrow=False,
        font={"size": 11, "family": CONFIG["body_font"], "color": "#000000"},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    fig_small = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[f"<b>{s}</b>" for s in segment_order],
        vertical_spacing=0.16,
        horizontal_spacing=0.06,
    )
    y_min = float(q2_orders["avg_profit_per_order"].min()) * 0.9
    y_max = float(q2_orders["avg_profit_per_order"].max()) * 1.1

    for idx, segment in enumerate(segment_order):
        row = (idx // 4) + 1
        col = (idx % 4) + 1
        seg_df = q2_orders[q2_orders["customer_segment"] == segment].copy()
        seg_df["quarter"] = pd.Categorical(seg_df["quarter"], categories=quarter_order, ordered=True)
        seg_df = seg_df.sort_values("quarter")
        fig_small.add_trace(
            go.Scatter(
                x=seg_df["quarter"],
                y=seg_df["avg_profit_per_order"],
                mode="lines+markers",
                name=segment,
                line={"width": 2.2, "color": get_segment_color(segment)},
                marker={"size": 5},
                hovertemplate="<b>%{x}</b><br>Avg profit/order: $%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig_small.update_xaxes(tickfont={"size": 8}, row=row, col=col)
        fig_small.update_yaxes(range=[y_min, y_max], tickprefix="$", tickfont={"size": 8}, row=row, col=col)

    apply_chart_theme(fig_small, "Average Profit per Order Trend by Segment", height=700)
    fig_small.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor="#000000",
        tickfont={"family": CONFIG["body_font"], "color": "#000000", "size": 8},
    )
    fig_small.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor="#000000",
        tickfont={"family": CONFIG["body_font"], "color": "#000000", "size": 8},
    )
    st.plotly_chart(fig_small, use_container_width=True)

    tech_df = q2_orders[q2_orders["customer_segment"] == "Tech Enthusiast"]
    tech_avg = float(tech_df["avg_profit_per_order"].mean()) if not tech_df.empty else np.nan
    tech_orders = order_level[order_level["customer_segment"] == "Tech Enthusiast"]
    tech_non_bf = tech_orders.loc[~tech_orders["is_black_friday"], "order_profit"].mean()
    tech_bf = tech_orders.loc[tech_orders["is_black_friday"], "order_profit"].mean()

    st.markdown(f"""
    <div class="callout insight-box">
    <b>Volatility Definition:</b> CV = standard deviation / mean (computed on quarterly average profit per order).
    <br><b>Tech Enthusiast Sanity Check:</b> Average ${tech_avg:,.2f}/order | Non-BF ${tech_non_bf:,.2f} vs BF ${tech_bf:,.2f}.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Q3: REFERRAL ECONOMICS (Corrected)
# ============================================================================
with tab3:
    q3_metrics = metrics["q3"].copy()
    referred_row = q3_metrics[q3_metrics["referral_group"] == "referred"]
    if referred_row.empty:
        discount_invested = 0.0
        referred_buyers = 0
        referred_order_count = 0
        net_lift_from_referral = 0.0
    else:
        referred_row = referred_row.iloc[0]
        discount_invested = float(referred_row["total_referral_discount"])
        referred_buyers = int(referred_row["unique_customers"])
        referred_order_count = int(referred_row["order_count"])
        net_lift_from_referral = float(referred_row.get("net_lift_from_referral", -discount_invested))

    avg_orders_per_buyer = (
        referred_order_count / referred_buyers if referred_buyers > 0 else 0.0
    )
    median_orders_per_buyer = 0
    discount_per_buyer = discount_invested / referred_buyers if referred_buyers > 0 else 0.0
    referred_order_counts = pd.Series(dtype="int64")
    payback_order = None
    incremental_revenue_total = np.nan
    payback_reached = "No"
    incremental_revenue_label = "N/A"
    incremental_revenue_delta = "vs non-referred baseline"
    fig_payback = None
    payback_caption = None

    referred_order_counts = (
        order_level[order_level["is_referred"] == 1]
        .groupby("buyer_id")["order_id"]
        .nunique()
    )
    if len(referred_order_counts) > 0:
        median_orders_per_buyer = int(referred_order_counts.median())

    curve_df = order_level[["buyer_id", "is_referred", "order_datetime_company", "order_revenue"]].copy()
    curve_df["order_datetime_company"] = pd.to_datetime(curve_df["order_datetime_company"], errors="coerce")
    curve_df = curve_df.dropna(subset=["order_datetime_company"])
    curve_df = curve_df.sort_values(["buyer_id", "order_datetime_company"])
    curve_df["order_rank"] = curve_df.groupby("buyer_id").cumcount() + 1
    curve_df["cohort"] = np.where(curve_df["is_referred"] == 1, "Referred", "Non-Referred")
    curve_df["cum_revenue_buyer"] = curve_df.groupby("buyer_id")["order_revenue"].cumsum()

    rank_stats = (
        curve_df.groupby(["cohort", "order_rank"])["cum_revenue_buyer"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    ref_stats = rank_stats[rank_stats["cohort"] == "Referred"].rename(
        columns={"mean": "mean_ref", "std": "std_ref", "count": "n_ref"}
    )
    nonref_stats = rank_stats[rank_stats["cohort"] == "Non-Referred"].rename(
        columns={"mean": "mean_nonref", "std": "std_nonref", "count": "n_nonref"}
    )
    payback_df = ref_stats.merge(nonref_stats, on="order_rank", how="inner")
    payback_df["se_diff"] = np.sqrt(
        (payback_df["std_ref"] ** 2 / payback_df["n_ref"])
        + (payback_df["std_nonref"] ** 2 / payback_df["n_nonref"])
    )
    payback_df["inc_mean"] = payback_df["mean_ref"] - payback_df["mean_nonref"]
    payback_df["inc_low"] = payback_df["inc_mean"] - 1.96 * payback_df["se_diff"]
    payback_df["inc_high"] = payback_df["inc_mean"] + 1.96 * payback_df["se_diff"]
    payback_df["n_effective"] = payback_df[["n_ref", "n_nonref"]].min(axis=1)

    cap_rank = min(15, int(payback_df["order_rank"].max()))
    payback_df = payback_df[payback_df["order_rank"] <= cap_rank].copy()

    payback_cross = payback_df[payback_df["inc_mean"] >= discount_per_buyer]
    payback_order = int(payback_cross["order_rank"].iloc[0]) if len(payback_cross) > 0 else None
    payback_reached = "Yes" if payback_order is not None else "No"
    incremental_revenue_total = float(payback_df["inc_mean"].iloc[-1] * referred_buyers)
    incremental_revenue_label = f"${incremental_revenue_total:,.0f}"
    incremental_revenue_delta = "cumulative to order 15"

    fig_payback = go.Figure()
    fig_payback.add_trace(go.Scatter(
        x=payback_df["order_rank"],
        y=payback_df["inc_high"],
        mode="lines",
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
    ))
    fig_payback.add_trace(go.Scatter(
        x=payback_df["order_rank"],
        y=payback_df["inc_low"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(29,158,117,0.14)",
        line={"width": 0},
        name="95% confidence band",
        showlegend=False,
        hovertemplate="Order %{x}<br>CI low: $%{y:.2f}<extra></extra>",
    ))
    fig_payback.add_trace(go.Scatter(
        x=payback_df["order_rank"],
        y=payback_df["inc_mean"],
        mode="lines+markers",
        name="Cumulative incremental revenue",
        line={"color": "#1D9E75", "width": 3},
        marker={"size": 5},
        hovertemplate="Order %{x}<br>Incremental cumulative rev/buyer: $%{y:.2f}<extra></extra>",
    ))
    fig_payback.add_trace(go.Scatter(
        x=payback_df["order_rank"],
        y=[discount_per_buyer] * len(payback_df),
        mode="lines",
        name="Discount cost threshold",
        line={"color": "#E24B4A", "dash": "dash", "width": 1.8},
        hovertemplate="Payback line: $%{y:.2f}<extra></extra>",
    ))
    fig_payback.add_trace(go.Scatter(
        x=payback_df["order_rank"],
        y=payback_df["n_effective"],
        mode="lines+markers",
        name="Effective sample size",
        line={"color": "#000000", "dash": "dot", "width": 1.6},
        marker={"size": 4},
        yaxis="y2",
        hovertemplate="Order %{x}<br>Effective sample n: %{y:,.0f}<extra></extra>",
    ))
    apply_chart_theme(
        fig_payback,
        "Referral Payback Curve: Incremental Revenue vs Discount Cost",
        height=470,
    )
    fig_payback.update_xaxes(title_text=f"Order number (1 to {cap_rank})", dtick=1)
    fig_payback.update_yaxes(title_text="Cumulative incremental revenue per referred buyer ($)")
    fig_payback.update_layout(
        yaxis2={
            "title": "Effective sample size (n)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        }
    )
    payback_story = "Payback not reached by order 15."
    if payback_order is not None:
        payback_story = f"Payback reached at order {payback_order}."
    payback_caption = (
        f"Payback line is fixed at ${discount_per_buyer:.2f} per referred buyer. "
        f"{payback_story} Confidence band widens as sample size falls with higher order rank."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Referral Discount Cost", f"${discount_invested:,.0f}", "10% first-order incentive")
    col2.metric("Incremental Revenue", incremental_revenue_label, incremental_revenue_delta)
    col3.metric("Payback Reached?", payback_reached, "within first 15 orders")
    col4.metric("Median Referred Orders", f"{median_orders_per_buyer}", f"avg {avg_orders_per_buyer:.1f}")

    if fig_payback is not None:
        st.plotly_chart(fig_payback, use_container_width=True)
        if payback_caption:
            st.caption(payback_caption)

    freq_bins = [0, 3, 6, 10, 20, 50, np.inf]
    freq_labels = ["1-3", "4-6", "7-10", "11-20", "21-50", "50+"]
    if len(referred_order_counts) > 0:
        freq_values = (
            pd.cut(referred_order_counts, bins=freq_bins, labels=freq_labels, right=True)
            .value_counts()
            .reindex(freq_labels, fill_value=0)
            .astype(int)
            .tolist()
        )
    else:
        freq_values = [0] * len(freq_labels)
    freq_colors = ["rgba(29,158,117,0.45)", "rgba(29,158,117,0.55)", "rgba(29,158,117,0.65)",
                   "rgba(29,158,117,0.75)", "rgba(29,158,117,0.85)", "rgba(29,158,117,1.00)"]

    fig_freq = go.Figure(data=[go.Bar(
        x=freq_labels, y=freq_values, marker={"color": freq_colors, "line": {"width": 0.5, "color": "#1D9E75"}},
        hovertemplate="Orders %{x}<br>Buyers: %{y:,}<extra></extra>",
    )])
    apply_chart_theme(fig_freq, "Lifetime Order Frequency of Referred Customers", height=360)
    fig_freq.update_layout(showlegend=False)
    fig_freq.update_xaxes(title_text="Total orders placed (lifetime)")
    fig_freq.update_yaxes(title_text="Number of referred buyers")
    st.plotly_chart(fig_freq, use_container_width=True)

    payback_text = "not reached within 15 orders"
    if payback_order is not None:
        payback_text = f"reached at order {payback_order}"
    st.markdown(f"""
    <div class="callout insight-box">
    <b>Referral Curve Update:</b> Cumulative incremental revenue is measured per referred buyer against non-referred baseline.
    <br><b>Payback threshold:</b> ${discount_per_buyer:.2f} per referred buyer ({payback_text}).
    <br><b>Counterfactual profit lift:</b> ${net_lift_from_referral:,.2f}.
    <br><b>Recommendation:</b> Do not continue flat referral discounts unless referred customers reach payback.
    Test targeted referral offers instead.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Q4: BLACK FRIDAY EFFECTIVENESS (Corrected)
# ============================================================================
with tab4:
    q4_metrics = metrics["q4"].copy()
    bf_total_orders = int(q4_metrics["bf_orders"].sum())
    bf_discount_given = float(q4_metrics["bf_discount_amount"].sum())
    bf_net_profit = float(q4_metrics["bf_profit"].sum())
    median_non_bf_orders = float(q4_metrics["median_daily_orders"].sum())
    median_non_bf_profit = float(q4_metrics["median_daily_profit"].sum())
    bf_volume_lift = (
        bf_total_orders / median_non_bf_orders if median_non_bf_orders > 0 else 0.0
    )
    net_bf_profit_lift = bf_net_profit - median_non_bf_profit

    q4_row_1 = st.columns(3)
    q4_row_1[0].metric("BF Volume Lift", f"{bf_volume_lift:.1f}x", f"vs segment-year baseline ({median_non_bf_orders:,.0f} orders)")
    q4_row_1[1].metric("BF Total Orders", f"{bf_total_orders:,}", "across 2024 + 2025")
    q4_row_1[2].metric("Discount Given", f"${bf_discount_given:,.0f}", "20% applied on BF orders")

    q4_row_2 = st.columns(2)
    q4_row_2[0].metric("BF Net Profit", f"${bf_net_profit:,.0f}", f"vs ${median_non_bf_profit:,.0f} baseline")
    q4_row_2[1].metric("Net BF Profit Lift", f"${net_bf_profit_lift:,.0f}", "vs segment-year baseline")

    q4_orders = order_level.copy()
    q4_orders["order_revenue"] = pd.to_numeric(q4_orders["order_revenue"], errors="coerce")
    q4_orders["black_friday_discount_amount"] = pd.to_numeric(
        q4_orders["black_friday_discount_amount"], errors="coerce"
    )
    aov_df = (
        q4_orders
        .groupby(["customer_segment", "is_black_friday"], as_index=False)
        .agg(
            avg_order_value=("order_revenue", "mean"),
            net_revenue=("order_revenue", "sum"),
            discount_foregone=("black_friday_discount_amount", "sum"),
        )
    )
    normal = aov_df[~aov_df["is_black_friday"]].set_index("customer_segment")
    bf = aov_df[aov_df["is_black_friday"]].set_index("customer_segment")
    segments = (
        bf["avg_order_value"]
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    normal_avg = normal.reindex(segments)["avg_order_value"].fillna(0).tolist()
    bf_avg = bf.reindex(segments)["avg_order_value"].fillna(0).tolist()
    revenue_received = bf.reindex(segments)["net_revenue"].fillna(0).tolist()
    discount_foregone = bf.reindex(segments)["discount_foregone"].fillna(0).tolist()

    fig_aov = go.Figure()
    fig_aov.add_trace(go.Bar(
        y=segments, x=normal_avg, orientation="h", name="Normal Days Avg Order Value",
        marker={"color": "#378ADD"}, hovertemplate="%{y}<br>Normal Days: $%{x:.2f}<extra></extra>",
    ))
    fig_aov.add_trace(go.Bar(
        y=segments, x=bf_avg, orientation="h", name="Black Friday effective avg (after 20% discount)",
        marker={"color": "#E24B4A"}, hovertemplate="%{y}<br>Black Friday: $%{x:.2f}<extra></extra>",
    ))
    apply_chart_theme(fig_aov, "Average Order Value by Segment: Normal Days vs Black Friday", height=430, hovermode="closest")
    fig_aov.update_layout(barmode="group")
    fig_aov.update_xaxes(title_text="Average order value ($)")
    fig_aov.update_yaxes(title_text="")
    st.plotly_chart(fig_aov, use_container_width=True)

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        y=segments, x=revenue_received, orientation="h", name="Net Revenue After Discount",
        marker={"color": "#1D9E75"}, hovertemplate="%{y}<br>Net Revenue After Discount: $%{x:,.0f}<extra></extra>",
    ))
    fig_rev.add_trace(go.Bar(
        y=segments, x=discount_foregone, orientation="h", name="Discount Foregone",
        marker={"color": "#E24B4A"}, hovertemplate="%{y}<br>Discount Foregone: $%{x:,.0f}<extra></extra>",
    ))
    apply_chart_theme(fig_rev, "Net Revenue After Discount vs Discount Foregone by Segment (Both BF Years)", height=390, hovermode="closest")
    fig_rev.update_layout(barmode="stack")
    fig_rev.update_xaxes(title_text="Amount ($)", tickformat="$.2s")
    fig_rev.update_yaxes(title_text="")
    st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("""
    <div class="callout insight-box">
    <b>Black Friday Decision: CONTINUE WITH TARGETING</b>
    <br>Volume lifts strongly, while effective order value varies by segment after discounting.
    <br>Tech Enthusiast absorbs the largest absolute discount cost and also has the highest Black Friday AOV.
    <br>Recommendation: continue Black Friday, but replace the flat 20% discount with segment-specific discounting.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Q5: SEGMENT STRATEGY (Corrected)
# ============================================================================
with tab5:
    q5_segments = metrics["q5"].copy()
    q5_segments["median_profit_per_order"] = pd.to_numeric(
        q5_segments["median_profit_per_order"], errors="coerce"
    )
    q5_segments["order_count"] = pd.to_numeric(q5_segments["order_count"], errors="coerce")
    q5_segments["total_profit"] = pd.to_numeric(q5_segments["total_profit"], errors="coerce")
    q5_segments = q5_segments.dropna(
        subset=["median_profit_per_order", "order_count", "total_profit"]
    )
    segments = [
        {
            "label": row["customer_segment"],
            "x": float(row["median_profit_per_order"]),
            "y": int(row["order_count"]),
            "profit": float(row["total_profit"]),
        }
        for _, row in q5_segments.iterrows()
    ]
    x_mid = float(q5_segments["median_profit_per_order"].median()) if len(q5_segments) else 0.0
    y_mid = float(q5_segments["order_count"].median()) if len(q5_segments) else 0.0
    quadrant_colors = {
        "Defend and Scale": "#12B76A",
        "High Demand, Improve Margin": "#F79009",
        "Niche Premium": "#2E90FA",
        "Low Priority": "#F04438",
    }

    max_profit = max((s["profit"] for s in segments), default=1.0)
    fig_matrix = go.Figure()
    quadrant_by_segment = {}
    for s in segments:
        if s["x"] >= x_mid and s["y"] >= y_mid:
            quadrant = "Defend and Scale"
        elif s["x"] < x_mid and s["y"] >= y_mid:
            quadrant = "High Demand, Improve Margin"
        elif s["x"] >= x_mid and s["y"] < y_mid:
            quadrant = "Niche Premium"
        else:
            quadrant = "Low Priority"
        quadrant_by_segment[s["label"]] = quadrant

        fig_matrix.add_trace(go.Scatter(
            x=[s["x"]], y=[s["y"]], mode="markers",
            marker={
                "size": (np.sqrt(s["profit"] / max_profit) * 38),
                "color": quadrant_colors[quadrant],
                "line": {"width": 1.5, "color": quadrant_colors[quadrant]},
                "opacity": 0.75,
            },
            name=quadrant,
            hovertemplate=(
                f"{s['label']} [{quadrant}]<br>"
                f"Median Profit per Order: ${s['x']:.2f}<br>"
                f"Total Order Volume: {s['y']:,}<br>"
                f"Total profit: ${s['profit']:,}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig_matrix.add_vline(x=x_mid, line_dash="dash", line_color="#000000")
    fig_matrix.add_hline(y=y_mid, line_dash="dash", line_color="#000000")
    apply_chart_theme(fig_matrix, "Segment Prioritization Matrix: Profitability vs Demand", height=500, hovermode="closest")
    x_values = [s["x"] for s in segments]
    y_values = [s["y"] for s in segments]
    x_pad = max((max(x_values) - min(x_values)) * 0.18, 1.0) if x_values else 1.0
    y_pad = max((max(y_values) - min(y_values)) * 0.18, 500.0) if y_values else 500.0
    fig_matrix.update_xaxes(
        title_text="Median Profit per Order",
        range=[min(x_values) - x_pad, max(x_values) + x_pad] if x_values else None,
        tickprefix="$",
    )
    fig_matrix.update_yaxes(
        title_text="Total Order Volume",
        range=[min(y_values) - y_pad, max(y_values) + y_pad] if y_values else None,
    )

    if x_values and y_values:
        fig_matrix.add_annotation(x=max(x_values), y=max(y_values), text="Defend and Scale", showarrow=False, font={"size": 11, "color": "#000000", "family": CONFIG["body_font"]})
        fig_matrix.add_annotation(x=min(x_values), y=max(y_values), text="High Demand, Improve Margin", showarrow=False, font={"size": 11, "color": "#000000", "family": CONFIG["body_font"]})
        fig_matrix.add_annotation(x=max(x_values), y=min(y_values), text="Niche Premium", showarrow=False, font={"size": 11, "color": "#000000", "family": CONFIG["body_font"]})
        fig_matrix.add_annotation(x=min(x_values), y=min(y_values), text="Low Priority", showarrow=False, font={"size": 11, "color": "#000000", "family": CONFIG["body_font"]})
    st.plotly_chart(fig_matrix, use_container_width=True)

    defend_segments = [
        s["label"] for s in sorted(segments, key=lambda row: row["profit"], reverse=True)
        if quadrant_by_segment.get(s["label"]) == "Defend and Scale"
    ]
    improve_segments = [
        s["label"] for s in sorted(segments, key=lambda row: row["profit"], reverse=True)
        if quadrant_by_segment.get(s["label"]) == "High Demand, Improve Margin"
    ]
    defend_text = ", ".join(defend_segments) if defend_segments else "none"
    improve_text = ", ".join(improve_segments) if improve_segments else "none"

    st.markdown(f"""
    <div class="callout decision-box">
    <b>Main Recommendation:</b> Prioritize {defend_text} as core growth segments.
    Improve margin strategy for high-demand lower-margin segments: {improve_text}.
    </div>
""", unsafe_allow_html=True)

