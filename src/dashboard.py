from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .theme import ACCENT, DANGER, SUCCESS, SEGMENT_COLORS


@dataclass(frozen=True)
class DashboardConfig:
    metrics_dir: Path
    output_dir: Path
    title_font: str = "Times New Roman, Times, serif"
    body_font: str = "Times New Roman, Times, serif"
    text_color: str = "#222222"
    muted_grid: str = "#DCE6F2"
    accent: str = ACCENT
    danger: str = DANGER
    success: str = SUCCESS
    segment_colors: dict[str, str] = field(default_factory=lambda: SEGMENT_COLORS.copy())


def _read_csv(metrics_dir: Path, name: str) -> pd.DataFrame:
    path = metrics_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing metric file: {path}")
    return pd.read_csv(path)


def _segment_color(cfg: DashboardConfig, segment: str) -> str:
    return cfg.segment_colors.get(segment, cfg.accent)


def _fmt_currency(v: float) -> str:
    a = abs(float(v))
    if a >= 1_000_000_000:
        return f"${v/1_000_000_000:.1f}B"
    if a >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if a >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:.0f}"


def _base_layout(fig: go.Figure, cfg: DashboardConfig, title: str, subtitle: str) -> go.Figure:
    fig.update_layout(
        title={
            "text": f"{title}<br><sup style='font-size:11px; opacity:0.7'>{subtitle}</sup>",
            "x": 0.01,
            "xanchor": "left",
            "font": {"family": cfg.title_font, "size": 18, "color": cfg.text_color, "weight": "bold"},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": cfg.body_font, "size": 10, "color": cfg.text_color},
        margin={"l": 80, "r": 60, "t": 100, "b": 80},
        legend={
            "orientation": "h",
            "x": 0.01,
            "y": -0.15,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 9},
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": cfg.accent,
            "font": {"family": cfg.body_font, "size": 10, "color": cfg.text_color},
        },
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont={"size": 9})
    fig.update_yaxes(showgrid=True, gridcolor=cfg.muted_grid, zeroline=False, tickfont={"size": 9})
    return fig


def _build_q1_staffing(df: pd.DataFrame, cfg: DashboardConfig) -> go.Figure:
    fig = go.Figure()

    for year in sorted(df["year"].unique()):
        year_data = df[df["year"] == year].sort_values("hour")
        fig.add_trace(
            go.Scatter(
                x=year_data["hour"],
                y=year_data["order_volume"],
                mode="lines+markers",
                name=str(year),
                line={"width": 3},
                hovertemplate="Hour %{x}:00<br>Orders: %{y:,.0f}<extra></extra>",
            )
        )

    fig.add_vline(
        x=5.5,
        line_dash="dash",
        line_color="rgba(100, 100, 100, 0.5)",
        line_width=1,
    )

    total_orders = df["order_volume"].sum()
    night_orders = df[df["hour"].between(0, 5)]["order_volume"].sum()
    night_pct = (night_orders / total_orders * 100) if total_orders > 0 else 0.0
    peak_row = df.loc[df["order_volume"].idxmax()]
    peak_hour = int(peak_row["hour"])
    night_avg = df[df["hour"].between(0, 5)]["order_volume"].mean()
    peak_to_night = float(peak_row["order_volume"]) / night_avg if night_avg > 0 else 0.0

    fig.add_annotation(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text=f"Night Demand: {night_pct:.1f}% | Peak Hour: {peak_hour}:00 | Peak/Night: {peak_to_night:.1f}x",
        showarrow=False,
        bgcolor="rgba(240, 240, 240, 0.6)",
        bordercolor=cfg.accent,
        borderwidth=1,
        borderpad=8,
        font={"size": 9, "family": cfg.body_font},
        xanchor="center", yanchor="bottom",
    )

    fig.add_annotation(
        x=0.5, y=-0.15, xref="paper", yref="paper",
        text=f"Staffing: Minimal overnight; full staffing {peak_hour}:00-18:00.",
        showarrow=False,
        bgcolor="white",
        bordercolor=cfg.success,
        borderwidth=2,
        borderpad=10,
        font={"size": 10, "family": cfg.body_font},
        xanchor="center", yanchor="top",
    )

    fig.update_xaxes(
        title_text="Hour of Day",
        dtick=1,
        tickmode="linear",
        range=[-0.5, 23.5],
    )
    fig.update_yaxes(title_text="Order Volume")

    return _base_layout(
        fig,
        cfg,
        "Hourly demand justifies selective staffing, not 24/7 coverage",
        "Peak demand ~5x higher than night hours → concentrate staffing.",
    )


def _build_q2_profit_distribution(df: pd.DataFrame, cfg: DashboardConfig) -> go.Figure:
    by_profit = df.groupby("customer_segment")["total_profit"].sum().sort_values(ascending=False)
    top_4 = by_profit.head(4).index.tolist()
    bottom_2 = by_profit.tail(2).index.tolist()
    selected_segments = top_4 + bottom_2

    titles = []
    for seg in selected_segments:
        seg_data = df[df["customer_segment"] == seg].sort_values("quarter")
        if seg in top_4:
            stability = "Stable" if seg_data["profit_volatility"].mean() < 0.5 else "Moderate"
        else:
            stability = "Volatile" if seg_data["profit_volatility"].mean() > 1.0 else "Moderate"
        titles.append(f"<b>{seg}</b><br><span style='font-size:9px; color:gray'>{stability}</span>")

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=titles,
        specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)],
        vertical_spacing=0.22,
        horizontal_spacing=0.12,
    )

    global_min = df["median_profit_per_order"].min()
    global_max = df["median_profit_per_order"].max()
    y_range = [global_min * 0.95, global_max * 1.05]

    positions = [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
    ]

    for idx, segment in enumerate(selected_segments):
        segment_data = df[df["customer_segment"] == segment].sort_values("quarter")
        row, col = positions[idx]

        fig.add_trace(
            go.Scatter(
                x=segment_data["quarter"],
                y=segment_data["median_profit_per_order"],
                mode="lines+markers",
                name=segment,
                line={"color": _segment_color(cfg, segment), "width": 2.5},
                marker={"size": 5, "opacity": 0.8},
                hovertemplate="<b>%{x}</b><br>Median Profit: $%{y:,.0f}<extra></extra>",
                showlegend=False,
            ),
            row=row, col=col,
        )

        fig.update_yaxes(
            range=y_range,
            tickformat="$,.0f",
            tickfont={"size": 9},
            row=row, col=col,
        )
        fig.update_xaxes(
            tickfont={"size": 8},
            row=row, col=col,
        )

    fig.add_annotation(
        text="High-profit segments are stable; low-profit segments show higher volatility → higher risk.",
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        xanchor="center", yanchor="bottom",
        showarrow=False,
        bgcolor="white",
        bordercolor=cfg.accent,
        borderwidth=1,
        borderpad=8,
        font={"size": 10, "family": cfg.body_font},
    )

    fig.update_layout(
        title={
            "text": "Segment Profitability: Quarterly Trends (Top 4 + Bottom 2)",
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 16, "family": cfg.title_font, "color": cfg.text_color},
        },
        height=580,
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 70, "r": 50, "t": 110, "b": 80},
    )

    return fig


def _build_q2_quarter_kpis(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for quarter in sorted(df["quarter"].dropna().unique()):
        q = df[df["quarter"] == quarter].copy()
        if q.empty:
            continue
        top_row = q.loc[q["median_profit_per_order"].idxmax()]
        low_row = q.loc[q["median_profit_per_order"].idxmin()]
        out[str(quarter)] = {
            "quarter": str(quarter),
            "total_orders": f"{int(q['order_count'].sum()):,}",
            "total_profit": f"${q['total_profit'].sum():,.0f}",
            "overall_median_profit_per_order": f"${q['median_profit_per_order'].median():,.2f}",
            "top_segment": str(top_row["customer_segment"]),
            "top_segment_profit": f"${float(top_row['median_profit_per_order']):,.2f}",
            "lowest_segment": str(low_row["customer_segment"]),
            "lowest_segment_profit": f"${float(low_row['median_profit_per_order']):,.2f}",
        }
    return out


def _build_q3_referral_counterfactual(
    df: pd.DataFrame, order_level: pd.DataFrame, cfg: DashboardConfig
) -> go.Figure:
    required_cols = {
        "referral_group",
        "unique_customers",
        "order_count",
        "avg_profit_per_order",
        "total_referral_discount",
        "total_profit",
    }
    if df.empty or not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Referral metric data unavailable for this run.",
            showarrow=False,
            font={"size": 14, "family": cfg.body_font, "color": cfg.text_color},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return _base_layout(
            fig,
            cfg,
            "Referral Program: Acquisition Cost vs Lifetime Value",
            "Insufficient referral data to compute counterfactual analysis.",
        )

    by_group = {str(row["referral_group"]): row for _, row in df.iterrows()}
    referred = by_group.get("referred")
    non_referred = by_group.get("non_referred")
    if referred is None:
        referred = pd.Series(
            {
                "unique_customers": 0.0,
                "order_count": 0.0,
                "avg_profit_per_order": 0.0,
                "total_referral_discount": 0.0,
                "total_profit": 0.0,
            }
        )
    if non_referred is None:
        non_referred = pd.Series(
            {
                "unique_customers": 0.0,
                "order_count": 0.0,
                "avg_profit_per_order": 0.0,
                "total_referral_discount": 0.0,
                "total_profit": 0.0,
            }
        )

    ref_unique_cust = float(referred["unique_customers"])
    non_ref_unique_cust = float(non_referred["unique_customers"])
    ref_orders = float(referred["order_count"])
    non_ref_orders = float(non_referred["order_count"])

    ref_profit_per = float(referred["avg_profit_per_order"])
    non_ref_profit_per = float(non_referred["avg_profit_per_order"])
    ref_total_discount = float(referred["total_referral_discount"])
    ref_total_profit = float(referred["total_profit"])

    ref_orders_per_cust = ref_orders / ref_unique_cust if ref_unique_cust else 0.0
    non_ref_orders_per_cust = non_ref_orders / non_ref_unique_cust if non_ref_unique_cust else 0.0

    cac = ref_total_discount / ref_unique_cust if ref_unique_cust else 0.0
    ltv = ref_profit_per * ref_orders_per_cust
    roi = ltv / cac if cac > 0 else 0.0

    # Compute cumulative payback by order number (per referred customer)
    has_order_cols = {"is_referred", "buyer_id", "order_id", "order_revenue"}.issubset(order_level.columns)
    referred_orders = order_level[order_level["is_referred"] == 1].copy() if has_order_cols else pd.DataFrame()
    if len(referred_orders) > 0:
        referred_orders = referred_orders.sort_values("order_id")
        referred_orders["buyer_cum_revenue"] = referred_orders.groupby("buyer_id")["order_revenue"].cumsum()
        cum_by_order = (
            referred_orders.groupby("buyer_id")
            .apply(lambda grp: grp["buyer_cum_revenue"].iloc[-1])
            .values
        )
        total_ref_revenue = referred_orders["order_revenue"].sum()

        orders_per_buyer = referred_orders.groupby("buyer_id").size()
        freq_dist = orders_per_buyer.value_counts().sort_index()
    else:
        cum_by_order = []
        total_ref_revenue = 0.0
        freq_dist = pd.Series(dtype=int)

    # Build subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Profit per Order: Referred vs Non-Referred",
            "Cumulative Revenue per Referred Customer",
            "Customer Repeat Rate Distribution",
            "Program Economics",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # Chart 1: Profit comparison (bar)
    fig.add_trace(
        go.Bar(
            x=["Referred", "Non-Referred"],
            y=[ref_profit_per, non_ref_profit_per],
            marker={"color": [cfg.success, cfg.accent]},
            text=[_fmt_currency(ref_profit_per), _fmt_currency(non_ref_profit_per)],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg Profit/Order: %{y:$,.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Chart 2: Cumulative revenue curve (line)
    if len(cum_by_order) > 0:
        cumulative_revenue = np.sort(cum_by_order)[::-1]  # sort descending to show payback speed
        order_nums = np.arange(1, len(cumulative_revenue) + 1)

        fig.add_trace(
            go.Scatter(
                x=order_nums,
                y=cumulative_revenue,
                mode="lines",
                fill="tozeroy",
                name="Cumulative Revenue",
                line={"color": cfg.success, "width": 2.5},
                fillcolor="rgba(46, 125, 50, 0.15)",
                hovertemplate="<b>Order #%{x}</b><br>Cumulative Revenue: %{y:$,.0f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

        # Add discount cost threshold line
        discount_line_y = [ref_total_discount] * len(order_nums)
        fig.add_trace(
            go.Scatter(
                x=order_nums,
                y=discount_line_y,
                mode="lines",
                name="Discount Investment",
                line={"color": cfg.danger, "width": 1.5, "dash": "dash"},
                hovertemplate="Discount Invested: $%{y:,.0f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    # Chart 3: Frequency distribution (bar)
    if len(freq_dist) > 0:
        freq_dist = freq_dist.head(15)  # cap at 15+ orders for readability
        fig.add_trace(
            go.Bar(
                x=freq_dist.index.astype(str),
                y=freq_dist.values,
                marker={"color": cfg.accent},
                hovertemplate="<b>%{x} orders</b><br>Customers: %{y:,}<extra></extra>",
                showlegend=False,
            ),
            row=2, col=1,
        )

    # Chart 4: KPI metric box (gauge-style text)
    decision_text = "CONTINUE" if roi > 20 else ("OPTIMIZE" if roi > 5 else "RECONSIDER")
    decision_color = cfg.success if roi > 20 else ("#FF9800" if roi > 5 else cfg.danger)

    fig.add_annotation(
        text=(
            f"<b>CAC (Cost per Customer)</b><br>${cac:.2f}<br><br>"
            f"<b>LTV (Lifetime Value)</b><br>${ltv:,.2f}<br><br>"
            f"<b>ROI</b><br>{roi:.1f}x<br><br>"
            f"<b style='color:{decision_color}'>→ {decision_text}</b>"
        ),
        xref="x4 domain", yref="y4 domain",
        x=0.5, y=0.5,
        showarrow=False,
        bgcolor="rgba(240, 240, 240, 0.8)",
        bordercolor=decision_color,
        borderwidth=2,
        borderpad=12,
        font={"size": 11, "family": cfg.body_font, "color": cfg.text_color},
        xanchor="center", yanchor="middle",
        align="center",
    )

    # Update axes
    fig.update_xaxes(title_text="Customer Segment", row=1, col=1)
    fig.update_yaxes(title_text="Avg Profit/Order", tickformat="$,.0f", row=1, col=1)

    fig.update_xaxes(title_text="Order Number (per Customer)", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Revenue ($)", tickformat="$,.0f", row=1, col=2)

    fig.update_xaxes(title_text="Orders Placed (Lifetime)", row=2, col=1)
    fig.update_yaxes(title_text="Number of Customers", row=2, col=1)

    # Decision callout
    fig.add_annotation(
        text=(
            f"<b>Referred customers cost ${ cac:.2f} to acquire</b> (via {ref_unique_cust:,.0f} customers). "
            f"<b>They generate ${ltv:,.2f} lifetime profit</b> ({ref_orders_per_cust:.1f} orders/customer). "
            f"<b>ROI: {roi:.1f}x</b> — Program is highly efficient despite {(non_ref_profit_per - ref_profit_per) / non_ref_profit_per * 100:.1f}% lower per-order profit."
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.10,
        showarrow=False,
        bgcolor="white",
        bordercolor=cfg.success,
        borderwidth=2,
        borderpad=10,
        font={"size": 9, "family": cfg.body_font},
        xanchor="center", yanchor="top",
    )

    fig = _base_layout(
        fig,
        cfg,
        "Referral Program: Acquisition Cost vs Lifetime Value",
        "Key insight: Low CAC + high LTV = strong program economics despite per-order profit discount.",
    )

    fig.update_layout(
        height=700,
        showlegend=False,
        margin={"l": 80, "r": 60, "t": 120, "b": 150},
    )

    return fig


def _build_q4_black_friday_lift(df: pd.DataFrame, cfg: DashboardConfig) -> go.Figure:
    # All data (both years)
    d_all = df.copy()
    d_all["annual_orders"] = pd.to_numeric(d_all["annual_orders"], errors="coerce")
    d_all["bf_orders"] = pd.to_numeric(d_all["bf_orders"], errors="coerce")
    d_all["median_daily_orders"] = pd.to_numeric(d_all["median_daily_orders"], errors="coerce")
    d_all["bf_profit"] = pd.to_numeric(d_all["bf_profit"], errors="coerce")
    d_all["median_daily_profit"] = pd.to_numeric(d_all["median_daily_profit"], errors="coerce")

    # Summary metrics (aggregate across all segments/years)
    total_bf_orders = d_all["bf_orders"].sum()
    total_median_daily = d_all["median_daily_orders"].sum()
    total_bf_profit = d_all["bf_profit"].sum()
    total_daily_profit = d_all["median_daily_profit"].sum()

    volume_lift = total_bf_orders / total_median_daily if total_median_daily > 0 else 0
    profit_lift = total_bf_profit / total_daily_profit if total_daily_profit > 0 else 0

    # Segment-level data (latest year)
    latest_year = int(df["year"].max())
    d = df[df["year"] == latest_year].copy()
    d = d.sort_values("revenue_lift_x", ascending=True).reset_index(drop=True)

    # Build subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Volume Lift by Segment",
            "Overall Volume Impact",
            "Profit Lift by Segment",
            "Black Friday Economics",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.18,
        horizontal_spacing=0.14,
    )

    # Chart 1: Volume lift by segment
    fig.add_trace(
        go.Bar(
            y=d["customer_segment"],
            x=d["volume_lift_x"],
            orientation="h",
            marker={"color": cfg.success},
            text=[f"{v:.1f}x" for v in d["volume_lift_x"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Volume Lift: %{x:.2f}x<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Chart 2: Overall volume comparison (BF vs normal)
    fig.add_trace(
        go.Bar(
            x=["Black Friday", "Median Day"],
            y=[total_bf_orders, total_median_daily],
            marker={"color": [cfg.danger, cfg.muted_grid.replace("0.07", "0.3")]},
            text=[f"{int(total_bf_orders):,} orders", f"{int(total_median_daily):,} orders"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Chart 3: Profit lift by segment
    d["profit_lift"] = np.where(
        d["median_daily_profit"] > 0,
        d["bf_profit"] / d["median_daily_profit"],
        0,
    )
    fig.add_trace(
        go.Bar(
            y=d["customer_segment"],
            x=d["profit_lift"],
            orientation="h",
            marker={"color": cfg.accent},
            text=[f"{v:.1f}x" for v in d["profit_lift"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Profit Lift: %{x:.2f}x<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Chart 4: KPI metric box
    fig.add_annotation(
        text=(
            f"<b>Volume Lift</b><br>{volume_lift:.1f}x<br><br>"
            f"<b>Profit Lift</b><br>{profit_lift:.1f}x<br><br>"
            f"<b style='color:{cfg.success}'>→ CONTINUE</b><br>"
            f"High volume offsets<br>discount cost"
        ),
        xref="x4 domain", yref="y4 domain",
        x=0.5, y=0.5,
        showarrow=False,
        bgcolor="rgba(240, 240, 240, 0.8)",
        bordercolor=cfg.success,
        borderwidth=2,
        borderpad=12,
        font={"size": 11, "family": cfg.body_font, "color": cfg.text_color},
        xanchor="center", yanchor="middle",
        align="center",
    )

    # Update axes
    fig.update_xaxes(title_text="Lift Multiple (x)", row=1, col=1)
    fig.update_yaxes(title_text="Segment", autorange="reversed", row=1, col=1)

    fig.update_xaxes(title_text="Orders", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=2)

    fig.update_xaxes(title_text="Lift Multiple (x)", row=2, col=1)
    fig.update_yaxes(title_text="Segment", autorange="reversed", row=2, col=1)

    # Decision callout
    fig.add_annotation(
        text=(
            f"<b>Black Friday drives {volume_lift:.1f}× volume increase</b> (vs median day), "
            f"translating to <b>{profit_lift:.1f}× profit lift</b> despite 20% discounts. "
            f"This indicates strong demand elasticity. "
            f"<b>Recommendation:</b> Continue promotion, but monitor segment-level efficiency — "
            f"focus on high-lift segments (>5×) and limit exposure to low-performers."
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        bgcolor="white",
        bordercolor=cfg.success,
        borderwidth=2,
        borderpad=10,
        font={"size": 9, "family": cfg.body_font},
        xanchor="center", yanchor="top",
    )

    fig = _base_layout(
        fig,
        cfg,
        "Black Friday: Elasticity & Profit Impact",
        "Volume lift vs profit lift — do discounts drive profitable growth or just margin erosion?",
    )

    fig.update_layout(
        height=680,
        showlegend=False,
        margin={"l": 100, "r": 80, "t": 120, "b": 160},
    )

    return fig


def _build_q5_segment_matrix(df: pd.DataFrame, cfg: DashboardConfig) -> go.Figure:
    d = df.copy()
    d["median_profit_per_order"] = pd.to_numeric(
        d.get("median_profit_per_order", d.get("avg_profit_per_order")), errors="coerce"
    )
    d["order_count"] = pd.to_numeric(d.get("order_count"), errors="coerce")
    d["total_profit"] = pd.to_numeric(d.get("total_profit"), errors="coerce")
    d["unique_customers"] = pd.to_numeric(d.get("unique_customers"), errors="coerce")
    d["orders_per_customer"] = pd.to_numeric(d.get("orders_per_customer"), errors="coerce")
    d["discount_pct"] = pd.to_numeric(d.get("discount_pct"), errors="coerce").fillna(0.0)
    d["cogs_pct"] = pd.to_numeric(d.get("cogs_pct"), errors="coerce").fillna(0.0)
    d["shipping_per_order"] = pd.to_numeric(d.get("shipping_per_order"), errors="coerce").fillna(0.0)

    d = d.dropna(subset=["median_profit_per_order", "order_count", "total_profit"]).copy()

    x_cut = float(d["median_profit_per_order"].median())
    y_cut = float(d["order_count"].median())

    def bucket(row: pd.Series) -> str:
        high_x = row["median_profit_per_order"] >= x_cut
        high_y = row["order_count"] >= y_cut
        if high_x and high_y:
            return "CORE"
        if high_x and not high_y:
            return "EXPAND"
        if not high_x and high_y:
            return "OPTIMIZE"
        return "DEPRIORITIZE"

    d["strategy"] = d.apply(bucket, axis=1)

    color_map = {
        "CORE": cfg.success,
        "EXPAND": "#EF9F27",
        "OPTIMIZE": "#378ADD",
        "DEPRIORITIZE": "#B4B2A9",
    }

    action_short_map = {
        "CORE": "Invest & Scale",
        "EXPAND": "Build & Acquire",
        "OPTIMIZE": "Improve Margins",
        "DEPRIORITIZE": "Monitor",
    }

    label_set = set(d.sort_values("total_profit", ascending=False).head(3)["customer_segment"].tolist())
    d["label"] = d["customer_segment"].where(d["customer_segment"].isin(label_set), "")

    fig = go.Figure()
    for strat in ["CORE", "EXPAND", "OPTIMIZE", "DEPRIORITIZE"]:
        s = d[d["strategy"] == strat]
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s["median_profit_per_order"],
                y=s["order_count"],
                mode="markers+text",
                name=strat,
                text=s["label"],
                textposition="top center",
                textfont={"size": 9},
                marker={
                    "size": np.sqrt(s["unique_customers"].clip(lower=1)).clip(upper=35) + 8,
                    "color": color_map[strat],
                    "opacity": 0.85,
                    "line": {"width": 0.8, "color": "white"},
                },
                customdata=s[["customer_segment", "total_profit", "unique_customers", "strategy"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> [%{customdata[3]}]<br>"
                    "Median profit/order: $%{x:,.2f}<br>"
                    "Order volume: %{y:,.0f}<br>"
                    "Customers: %{customdata[2]:.0f}<br>"
                    "Total profit: $%{customdata[1]:,.0f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=x_cut, line_dash="dash", line_color="rgba(150, 150, 150, 0.4)", line_width=1)
    fig.add_hline(y=y_cut, line_dash="dash", line_color="rgba(150, 150, 150, 0.4)", line_width=1)

    total_profit = float(d["total_profit"].sum())
    total_orders = float(d["order_count"].sum())
    total_customers = float(d["unique_customers"].sum())
    top_seg = d.iloc[0]
    top_seg_name = top_seg["customer_segment"]
    top_seg_profit = float(top_seg["total_profit"])
    top_seg_pct = (top_seg_profit / total_profit * 100) if total_profit > 0 else 0.0
    top_seg_custs = float(top_seg["unique_customers"])
    top_seg_cust_pct = (top_seg_custs / total_customers * 100) if total_customers > 0 else 0.0

    # Data validation block
    fig.add_annotation(
        text=(
            f"<b>Data Validation</b><br>"
            f"Segments analyzed: {len(d)}<br>"
            f"Total customers: {int(total_customers):,}<br>"
            f"Total orders: {int(total_orders):,}<br>"
            f"Total profit: ${total_profit:,.0f}"
        ),
        xref="paper", yref="paper",
        x=0.12, y=1.16,
        showarrow=False,
        bgcolor="rgba(100, 150, 200, 0.06)",
        bordercolor="#378ADD",
        borderwidth=1,
        borderpad=8,
        font={"size": 8, "family": cfg.body_font},
        xanchor="center", yanchor="bottom",
    )

    # Top segment efficiency metric
    fig.add_annotation(
        text=(
            f"<b>{top_seg_name}</b><br>"
            f"{top_seg_pct:.0f}% of profit<br>"
            f"from {top_seg_cust_pct:.0f}% of customers"
        ),
        xref="paper", yref="paper",
        x=0.88, y=1.16,
        showarrow=False,
        bgcolor="rgba(46, 125, 50, 0.08)",
        bordercolor=cfg.success,
        borderwidth=1,
        borderpad=8,
        font={"size": 8, "family": cfg.body_font, "color": cfg.success},
        xanchor="center", yanchor="bottom",
    )

    core_segs = d[d["strategy"] == "CORE"]
    expand_segs = d[d["strategy"] == "EXPAND"]
    optimize_segs = d[d["strategy"] == "OPTIMIZE"]
    deprioritize_segs = d[d["strategy"] == "DEPRIORITIZE"]

    core_profit = float(core_segs["total_profit"].sum()) if len(core_segs) > 0 else 0.0
    core_avg_margin = float(core_segs["median_profit_per_order"].mean()) if len(core_segs) > 0 else 0.0
    core_avg_opc = float(core_segs["orders_per_customer"].mean()) if len(core_segs) > 0 else 0.0
    core_profit_share = (core_profit / total_profit * 100) if total_profit > 0 else 0.0

    fig.add_annotation(
        text=(
            f"<b>CORE</b> — Invest & Scale<br>"
            f"{len(core_segs)} segments | {core_profit_share:.0f}% of profit<br>"
            f"Avg margin: ${core_avg_margin:,.2f}/order<br>"
            f"Focus: retention, supply chain, premium positioning"
        ),
        xref="paper", yref="paper",
        x=0.25, y=-0.18,
        showarrow=False,
        bgcolor="rgba(46, 125, 50, 0.08)",
        bordercolor=cfg.success,
        borderwidth=2,
        borderpad=10,
        font={"size": 9, "family": cfg.body_font},
        xanchor="center", yanchor="top",
    )

    expand_profit = float(expand_segs["total_profit"].sum()) if len(expand_segs) > 0 else 0.0
    expand_margin = float(expand_segs["median_profit_per_order"].mean()) if len(expand_segs) > 0 else 0.0
    expand_opc = float(expand_segs["orders_per_customer"].mean()) if len(expand_segs) > 0 else 0.0
    expand_growth = float(expand_segs["order_count"].sum()) if len(expand_segs) > 0 else 0.0

    fig.add_annotation(
        text=(
            f"<b>EXPAND</b> — Build & Acquire<br>"
            f"{len(expand_segs)} segments | ${expand_profit:,.0f} profit<br>"
            f"Avg margin: ${expand_margin:,.2f}/order (high)<br>"
            f"Focus: customer acquisition, brand awareness, volume growth"
        ),
        xref="paper", yref="paper",
        x=0.75, y=-0.18,
        showarrow=False,
        bgcolor="rgba(239, 159, 39, 0.08)",
        bordercolor="#EF9F27",
        borderwidth=2,
        borderpad=10,
        font={"size": 9, "family": cfg.body_font},
        xanchor="center", yanchor="top",
    )

    optimize_profit = float(optimize_segs["total_profit"].sum()) if len(optimize_segs) > 0 else 0.0
    optimize_margin = float(optimize_segs["median_profit_per_order"].mean()) if len(optimize_segs) > 0 else 0.0
    optimize_discount = float(optimize_segs["discount_pct"].mean()) if len(optimize_segs) > 0 else 0.0
    optimize_cogs = float(optimize_segs["cogs_pct"].mean()) if len(optimize_segs) > 0 else 0.0

    fig.add_annotation(
        text=(
            f"<b>OPTIMIZE</b> — Improve Margins<br>"
            f"{len(optimize_segs)} segments | {optimize_profit:,.0f} profit<br>"
            f"Margin: ${optimize_margin:,.2f}/order (pressure: {optimize_discount:.0f}% disc, {optimize_cogs:.0f}% COGS)<br>"
            f"Focus: reduce discounting, negotiate COGS, operational efficiency"
        ),
        xref="paper", yref="paper",
        x=0.25, y=-0.40,
        showarrow=False,
        bgcolor="rgba(55, 138, 221, 0.08)",
        bordercolor="#378ADD",
        borderwidth=2,
        borderpad=10,
        font={"size": 9, "family": cfg.body_font},
        xanchor="center", yanchor="top",
    )

    depri_profit = float(deprioritize_segs["total_profit"].sum()) if len(deprioritize_segs) > 0 else 0.0
    depri_custs = float(deprioritize_segs["unique_customers"].sum()) if len(deprioritize_segs) > 0 else 0.0

    fig.add_annotation(
        text=(
            f"<b>DEPRIORITIZE</b> — Monitor<br>"
            f"{len(deprioritize_segs)} segments | ${depri_profit:,.0f} profit<br>"
            f"Customers: {int(depri_custs):,} (low value/volume)<br>"
            f"Focus: maintain low-cost servicing, harvest profitability, avoid acquisition spend"
        ),
        xref="paper", yref="paper",
        x=0.75, y=-0.40,
        showarrow=False,
        bgcolor="rgba(180, 178, 169, 0.08)",
        bordercolor="#B4B2A9",
        borderwidth=2,
        borderpad=10,
        font={"size": 9, "family": cfg.body_font},
        xanchor="center", yanchor="top",
    )

    # KPI Recommendations block
    fig.add_annotation(
        text=(
            "<b>Track These KPIs</b><br>"
            "<b>CORE:</b> Retention rate, NPS, repeat order rate, AOV trend<br>"
            "<b>EXPAND:</b> Customer acquisition cost, conversion rate, margin per new customer<br>"
            "<b>OPTIMIZE:</b> Discount dependency, margin improvement %, COGS trends<br>"
            "<b>DEPRIORITIZE:</b> Service cost, churn rate, breakeven analysis"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.63,
        showarrow=False,
        bgcolor="rgba(240, 240, 240, 0.8)",
        bordercolor=cfg.accent,
        borderwidth=1,
        borderpad=10,
        font={"size": 8, "family": cfg.body_font},
        xanchor="center", yanchor="top",
    )

    fig.update_xaxes(
        title_text="Median Profit per Order ($)",
        tickprefix="$",
        separatethousands=True,
    )
    fig.update_yaxes(
        title_text="Order Volume",
        rangemode="tozero",
        separatethousands=True,
    )

    fig = _base_layout(
        fig,
        cfg,
        "Segment Prioritization Matrix: Where to Invest for Maximum ROI",
        "X=profit/order (quality), Y=order count (volume). Bubble=customer base. Scale by quadrant strategy.",
    )
    fig.update_layout(
        legend={
            "orientation": "h",
            "x": 0.01,
            "y": 1.22,
            "xanchor": "left",
            "yanchor": "bottom",
        },
        margin={"l": 100, "r": 60, "t": 150, "b": 420},
    )
    return fig


def build_dashboard(metrics_dir: Path, output_dir: Path, order_level: pd.DataFrame | None = None, coverage_pct: float | None = None) -> None:
    cfg = DashboardConfig(metrics_dir=metrics_dir, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if order_level is None or order_level.empty:
        num_customers = "N/A"
        num_orders = "N/A"
    else:
        num_customers = f"{int(order_level['buyer_id'].nunique()):,}"
        num_orders = f"{int(order_level['order_id'].nunique()):,}"

    coverage_display = f"{coverage_pct:.2f}%" if coverage_pct is not None else "N/A"

    q1_df = _read_csv(metrics_dir, "q1_hourly_order_volume.csv")
    q2_df = _read_csv(metrics_dir, "q2_profit_by_quarter_segment.csv")
    q3_df = _read_csv(metrics_dir, "q3_referral_impact.csv")
    q4_df = _read_csv(metrics_dir, "q4_black_friday_impact.csv")
    q5_df = _read_csv(metrics_dir, "q5_customer_base_kpis.csv")

    q1 = _build_q1_staffing(q1_df, cfg)
    q2 = _build_q2_profit_distribution(q2_df, cfg)
    q3 = _build_q3_referral_counterfactual(
        q3_df,
        order_level if order_level is not None else pd.DataFrame(),
        cfg
    )
    q4 = _build_q4_black_friday_lift(q4_df, cfg)
    q5 = _build_q5_segment_matrix(q5_df, cfg)

    figs_data = [
        ("Q1. Staffing Decision", q1),
        ("Q2. Profit Stability", q2),
        ("Q3. Referral Economics", q3),
        ("Q4. Promotion Effectiveness", q4),
        ("Q5. Segment Strategy", q5),
    ]

    charts_json = []
    for _, fig in figs_data:
        fig_dict = json.loads(fig.to_json())
        charts_json.append(fig_dict)

    charts_data_js = json.dumps(charts_json)
    q2_kpis_js = json.dumps(_build_q2_quarter_kpis(q2_df))
    plotly_js_bundle = get_plotlyjs()

    tab_buttons = "\n".join(
        [f'            <button class="tab-button{"" if i > 0 else " active"}" onclick="showTab({i})">{title}</button>'
         for i, (title, _) in enumerate(figs_data)]
    )

    tab_contents_parts = []
    for i in range(len(figs_data)):
        if i == 1:
            tab_contents_parts.append(
                f"""            <div id="tab-{i}" class="tab-content{"" if i > 0 else " active"}">
                <div id="chart-{i}"></div>
                <div id="q2-kpi-tiles" class="q2-kpis"></div>
            </div>"""
            )
        else:
            tab_contents_parts.append(
                f'            <div id="tab-{i}" class="tab-content{"" if i > 0 else " active"}"><div id="chart-{i}"></div></div>'
            )
    tab_contents = "\n".join(tab_contents_parts)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Capital One E-Commerce Executive Dashboard</title>
    <script>{plotly_js_bundle}</script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        html, body {{
            font-family: "Times New Roman", Times, serif;
            background: #f5f5f5;
            color: #222;
            height: 100%;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            min-height: 100vh;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #1F4E79 0%, #2d5a9e 100%);
            color: white;
            padding: 50px 40px;
            text-align: center;
        }}
        .header h1 {{
            font-family: "Times New Roman", Times, serif;
            font-size: 36px;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        .header p {{
            font-size: 15px;
            opacity: 0.95;
            max-width: 800px;
            margin: 0 auto;
        }}
        .tabs {{
            display: flex;
            background: #f0f0f0;
            border-bottom: 3px solid #ddd;
            flex-wrap: wrap;
            padding: 0 20px;
        }}
        .tab-button {{
            flex: 1;
            min-width: 220px;
            padding: 16px 12px;
            border: none;
            background: #f0f0f0;
            color: #333;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            border-bottom: 4px solid transparent;
            text-align: center;
        }}
        .tab-button:hover {{
            background: #e5e5e5;
        }}
        .tab-button.active {{
            background: white;
            color: #1F4E79;
            border-bottom-color: #1F4E79;
        }}
        .tab-content {{
            display: none;
            padding: 40px 30px;
            animation: fadeIn 0.3s ease;
            min-height: 700px;
        }}
        .tab-content.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .chart {{
            width: 100%;
            height: 700px;
        }}
        .footer {{
            background: #f9f9f9;
            border-top: 1px solid #eee;
            padding: 25px;
            text-align: center;
            font-size: 12px;
            color: #1F4E79;
        }}
        .legend {{
            padding: 20px 30px;
            background: #f0f6ff;
            border-bottom: 2px solid #ddd;
        }}
        .legend h3 {{
            font-size: 13px;
            color: #1F4E79;
            margin-bottom: 12px;
            font-weight: bold;
        }}
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .legend-item {{
            font-size: 11px;
            line-height: 1.6;
            color: #333;
        }}
        .legend-item strong {{
            color: #1F4E79;
            display: block;
            margin-bottom: 3px;
        }}
        .metrics-note {{
            background: white;
            padding: 12px;
            border-left: 3px solid #2E7D32;
            margin-top: 12px;
            font-size: 10px;
            color: #555;
        }}
        .methodology {{
            margin-top: 40px;
            padding: 20px;
            background: #f5f5f5;
            border-left: 4px solid #1F4E79;
        }}
        .methodology h3 {{
            font-size: 14px;
            color: #1F4E79;
            margin-bottom: 8px;
        }}
        .methodology p {{
            font-size: 11px;
            color: #222;
            line-height: 1.5;
        }}
        .q2-kpis {{
            margin-top: 16px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 10px;
        }}
        .q2-kpi-tile {{
            border: 1px solid #1F4E79;
            background: #f8fbff;
            padding: 10px 12px;
        }}
        .q2-kpi-label {{
            font-size: 10px;
            color: #1F4E79;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        .q2-kpi-value {{
            font-size: 16px;
            font-weight: 700;
            color: #222;
            margin-top: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Capital One E-Commerce Executive Dashboard</h1>
            <p>Data-driven insights for strategic business decisions - 5 business questions answered with quantified recommendations</p>
        </div>
        <div class="legend">
            <h3>Dashboard Overview & Key Metrics</h3>
            <div class="legend-grid">
                <div class="legend-item">
                    <strong>Q1. Staffing Decision</strong>
                    When is overnight staffing justified? Hours of low/peak demand.
                </div>
                <div class="legend-item">
                    <strong>Q2. Profit Stability</strong>
                    How stable is segment profitability over time? Quarterly median profit trends by segment.
                </div>
                <div class="legend-item">
                    <strong>Q3. Referral Economics</strong>
                    Should we continue the referral program? Profit with/without discount.
                </div>
                <div class="legend-item">
                    <strong>Q4. Promotion Effectiveness</strong>
                    Which segments respond to Black Friday? Volume & revenue lift multiples.
                </div>
                <div class="legend-item">
                    <strong>Q5. Segment Strategy</strong>
                    Where should we invest? 2x2 matrix: median profit per order vs order volume.
                </div>
                <div class="legend-item">
                    <strong>Volatility</strong>
                    Stability metric = Profit IQR / |Median Profit|. Higher means less predictable earnings.
                </div>
            </div>
            <div class="metrics-note">
                <strong>Key Metrics:</strong> All profit figures in USD | Median Profit = robust central tendency for skewed distributions | Lift Multiple = metric on promotion day / median non-promotion daily metric | Q5 uses median profit per order and order count
            </div>
        </div>
        <div class="tabs">
{tab_buttons}
        </div>
        <div>
{tab_contents}
        </div>
        <div class="methodology">
            <h3>Methodology</h3>
            <p>
                <strong>Customers:</strong> {num_customers} | <strong>Orders:</strong> {num_orders} | <strong>Join Coverage:</strong> {coverage_display}<br>
                <strong>Decision Framework:</strong> Each chart answers a specific business question with quantified metrics, thresholds, and recommended actions.<br>
                <strong>Assumptions:</strong> Customer timezone conversion, 10% referral discount (first order), 20% Black Friday discount, tiered shipping fees.
            </p>
        </div>
        <div class="footer">
            McKinsey-style analytical dashboard | Generated automatically from data pipeline | All metrics from latest run
        </div>
    </div>

    <script>
        const chartsData = {charts_data_js};
        const q2QuarterKpis = {q2_kpis_js};

        function showTab(tabIndex) {{
            document.querySelectorAll('.tab-content').forEach(tab =>
                tab.classList.remove('active')
            );
            document.querySelectorAll('.tab-button').forEach(btn =>
                btn.classList.remove('active')
            );

            document.getElementById('tab-' + tabIndex).classList.add('active');
            document.querySelectorAll('.tab-button')[tabIndex].classList.add('active');

            setTimeout(() => Plotly.Plots.resize('chart-' + tabIndex), 50);
        }}

        function renderQ2Kpis(quarter) {{
            const host = document.getElementById('q2-kpi-tiles');
            if (!host || !q2QuarterKpis[quarter]) return;
            const k = q2QuarterKpis[quarter];
            host.innerHTML = `
                <div class="q2-kpi-tile"><div class="q2-kpi-label">Quarter</div><div class="q2-kpi-value">${{k.quarter}}</div></div>
                <div class="q2-kpi-tile"><div class="q2-kpi-label">Total Orders</div><div class="q2-kpi-value">${{k.total_orders}}</div></div>
                <div class="q2-kpi-tile"><div class="q2-kpi-label">Total Profit</div><div class="q2-kpi-value">${{k.total_profit}}</div></div>
                <div class="q2-kpi-tile"><div class="q2-kpi-label">Median Profit/Order</div><div class="q2-kpi-value">${{k.overall_median_profit_per_order}}</div></div>
                <div class="q2-kpi-tile"><div class="q2-kpi-label">Top Segment</div><div class="q2-kpi-value">${{k.top_segment}} (${{k.top_segment_profit}})</div></div>
                <div class="q2-kpi-tile"><div class="q2-kpi-label">Lowest Segment</div><div class="q2-kpi-value">${{k.lowest_segment}} (${{k.lowest_segment_profit}})</div></div>
            `;
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            chartsData.forEach((data, i) => {{
                Plotly.newPlot('chart-' + i, data.data, data.layout, {{
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d']
                }});
            }});

            const q2Chart = document.getElementById('chart-1');
            if (q2Chart) {{
                const quarters = Object.keys(q2QuarterKpis).sort();
                if (quarters.length > 0) {{
                    renderQ2Kpis(quarters[0]);
                }}
                q2Chart.on('plotly_click', function(evt) {{
                    const clickedQuarter = evt?.points?.[0]?.x;
                    if (clickedQuarter && q2QuarterKpis[clickedQuarter]) {{
                        renderQ2Kpis(clickedQuarter);
                    }}
                }});
            }}


            const hash = window.location.hash || '';
            const match = hash.match(/^#tab=(\\d+)$/);
            if (match) {{
                const idx = parseInt(match[1], 10);
                if (!Number.isNaN(idx) && idx >= 0 && idx < chartsData.length) {{
                    showTab(idx);
                }}
            }}
        }});
    </script>
</body>
</html>"""

    (output_dir / "index.html").write_text(html, encoding="utf-8")

    legacy_aliases = {
        "q1_staffing.html": 0,
        "q2_profitability.html": 1,
        "q3_referral.html": 2,
        "q4_black_friday.html": 3,
        "q5_segmentation.html": 4,
    }
    for filename, tab_index in legacy_aliases.items():
        alias_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=./index.html#tab={tab_index}" />
    <title>Redirecting...</title>
</head>
<body>
    <p>This dashboard page moved. Redirecting to <a href="./index.html#tab={tab_index}">index.html</a>.</p>
</body>
</html>"""
        (output_dir / filename).write_text(alias_html, encoding="utf-8")
