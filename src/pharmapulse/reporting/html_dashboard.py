"""
Builds `reports/pharmapulse_dashboard.html`: a single, self-contained,
interactive HTML dashboard (Plotly charts + a styled KPI/table layout)
that a supply-chain or finance manager could open directly in a
browser - no server, no external dependencies (Plotly.js is embedded
inline once, so the file works fully offline).

It is built from exactly the same computed tables that populate the
Excel workbook (`scripts/run_pipeline.py`), so the two artifacts are
two views onto one source of truth rather than two independently
computed reports.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Red/orange theme
# ---------------------------------------------------------------------------
PALETTE = dict(
    primary="#C0392B",       # deep red - primary series
    secondary="#E67E22",     # warm orange - secondary series
    accent="#F5B041",        # amber/gold - highlights
    deep="#7B241C",          # dark brick red - headers, emphasis
    warn="#CB4335",          # alert red
    ink="#4A2A20",           # warm dark brown-red for text
    grid="#F3E0D6",
    band="rgba(230,126,34,0.18)",
    good="#27AE60",          # kept green only for "good/on-target" signals
)

CHART_COLORWAY = ["#C0392B", "#E67E22", "#F5B041", "#922B21", "#EB984E", "#A04000"]


def _fig_to_div(fig: go.Figure, div_id: str, include_js: str | bool = False) -> str:
    fig.update_layout(
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=PALETTE["ink"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=CHART_COLORWAY,
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"])
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_js, div_id=div_id)


# ---------------------------------------------------------------------------
# Charts: portfolio & service level
# ---------------------------------------------------------------------------
def _revenue_chart(revenue_by_category: pd.DataFrame) -> go.Figure:
    df = revenue_by_category.sort_values("total_revenue")
    fig = go.Figure(go.Bar(
        x=df["total_revenue"], y=df["category"], orientation="h",
        marker_color=PALETTE["primary"],
        text=[f"${v:,.0f}" for v in df["total_revenue"]], textposition="outside",
    ))
    fig.update_layout(title="Revenue by Therapeutic Category", height=380)
    return fig


def _fill_rate_chart(fill_rate_by_month: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=fill_rate_by_month["month"], y=fill_rate_by_month["fill_rate"],
        mode="lines+markers", line=dict(color=PALETTE["secondary"], width=3),
        fill="tozeroy", fillcolor="rgba(230,126,34,0.10)",
    ))
    fig.add_hline(y=0.95, line_dash="dash", line_color=PALETTE["good"],
                  annotation_text="95% target service level")
    fig.update_layout(title="Network Fill Rate Trend", yaxis_tickformat=".0%", height=380,
                       yaxis_range=[0.5, 1.02])
    return fig


def _forecast_chart(history: pd.DataFrame, future: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["week_start"], y=history["weekly_units"], mode="lines",
        name="Observed (history)", line=dict(color=PALETTE["ink"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([future["week_start"], future["week_start"][::-1]]),
        y=pd.concat([future["forecast_p95"], future["forecast_p50"][::-1]]),
        fill="toself", fillcolor=PALETTE["band"], line=dict(color="rgba(0,0,0,0)"),
        name="P50-P95 forecast band", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=future["week_start"], y=future["forecast_p50"], mode="lines+markers",
        name="Forecast (P50)", line=dict(color=PALETTE["primary"], width=3, dash="dash"),
    ))
    fig.update_layout(title="Network-wide Weekly Demand: History & 12-Week Forecast", height=420)
    return fig


def _abc_xyz_heatmap(abc_xyz: pd.DataFrame) -> go.Figure:
    pivot = abc_xyz.pivot_table(
        index="abc_class", columns="xyz_class", values="total_revenue", aggfunc="sum", fill_value=0
    ).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{c} (variability)" for c in pivot.columns],
        y=[f"{r} (revenue tier)" for r in pivot.index],
        colorscale=[[0, "#FDF2E9"], [0.5, "#F0B27A"], [1, "#7B241C"]],
        text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}", hoverinfo="skip",
    ))
    fig.update_layout(title="ABC × XYZ Segmentation (revenue by cell)", height=380)
    return fig


def _category_bar_counts(abc_xyz: pd.DataFrame) -> go.Figure:
    counts = abc_xyz["segment"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=PALETTE["accent"]))
    fig.update_layout(title="SKU-Center Pairs per Segment", height=380)
    return fig


# ---------------------------------------------------------------------------
# Charts: working capital (CCC)
# ---------------------------------------------------------------------------
def _ccc_gauge(ccc_summary: dict) -> go.Figure:
    ccc = ccc_summary["CCC_days"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ccc,
        number={"suffix": " days", "font": {"color": PALETTE["deep"], "size": 40}},
        delta={"reference": 60, "increasing": {"color": PALETTE["warn"]}, "decreasing": {"color": PALETTE["good"]}},
        gauge={
            "axis": {"range": [0, 120], "tickcolor": PALETTE["ink"]},
            "bar": {"color": PALETTE["primary"]},
            "steps": [
                {"range": [0, 45], "color": "#FDEBD0"},
                {"range": [45, 75], "color": "#F5CBA7"},
                {"range": [75, 120], "color": "#EDBB99"},
            ],
            "threshold": {"line": {"color": PALETTE["deep"], "width": 4}, "thickness": 0.85, "value": 60},
        },
        title={"text": "Cash Conversion Cycle (60-day trailing)", "font": {"size": 15}},
    ))
    fig.update_layout(height=340)
    return fig


def _ccc_components_chart(ccc_summary: dict) -> go.Figure:
    labels = ["DIO<br>(Inventory)", "DSO<br>(Receivables)", "DPO<br>(Payables)"]
    values = [ccc_summary["DIO_days"], ccc_summary["DSO_days"], -ccc_summary["DPO_days"]]
    colors = [PALETTE["secondary"], PALETTE["primary"], PALETTE["good"]]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"+{ccc_summary['DIO_days']:.0f}d", f"+{ccc_summary['DSO_days']:.0f}d", f"-{ccc_summary['DPO_days']:.0f}d"],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color=PALETTE["ink"], line_width=1)
    fig.update_layout(title="CCC Components (DIO + DSO − DPO)", height=340, showlegend=False)
    return fig


def _ccc_trend_chart(wc_df: pd.DataFrame) -> go.Figure:
    df = wc_df.dropna(subset=["CCC"]).tail(365)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["DIO"], name="DIO", line=dict(color=PALETTE["secondary"], width=2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["DSO"], name="DSO", line=dict(color=PALETTE["primary"], width=2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["DPO"], name="DPO", line=dict(color=PALETTE["good"], width=2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["CCC"], name="CCC", line=dict(color=PALETTE["deep"], width=3, dash="dot")))
    fig.add_hline(y=60, line_dash="dash", line_color=PALETTE["ink"], annotation_text="60-day CCC target")
    fig.update_layout(title="Working Capital Trend (trailing 60-day window)", yaxis_title="Days", height=400)
    return fig


# ---------------------------------------------------------------------------
# Charts: raw material supply & production
# ---------------------------------------------------------------------------
def _supplier_ontime_chart(supplier_performance: pd.DataFrame) -> go.Figure:
    df = supplier_performance.sort_values("on_time_rate")
    colors = [PALETTE["warn"] if r < 0.85 else PALETTE["secondary"] for r in df["on_time_rate"]]
    fig = go.Figure(go.Bar(
        x=df["on_time_rate"], y=df["supplier_name"], orientation="h", marker_color=colors,
        text=[f"{r:.0%}" for r in df["on_time_rate"]], textposition="outside",
    ))
    fig.add_vline(x=0.90, line_dash="dash", line_color=PALETTE["ink"], annotation_text="90% target")
    fig.update_layout(title="Supplier On-Time Delivery Rate", xaxis_tickformat=".0%", height=380)
    return fig


def _material_coverage_chart(material_coverage: pd.DataFrame, top_n: int = 12) -> go.Figure:
    df = material_coverage.sort_values("days_of_cover").head(top_n)
    colors = [PALETTE["warn"] if r else PALETTE["accent"] for r in df["at_risk"]]
    fig = go.Figure(go.Bar(
        x=df["days_of_cover"], y=df["material_name"], orientation="h", marker_color=colors,
        text=[f"{v:.0f}d (lead time {lt:.0f}d)" for v, lt in zip(df["days_of_cover"], df["avg_lead_time_days"])],
        textposition="outside",
    ))
    fig.update_layout(title="Lowest Raw-Material Days of Cover vs. Supplier Lead Time", height=420)
    return fig


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def _kpi_cards_html(kpis: list[tuple[str, str, str]]) -> str:
    cards = []
    for label, value, sub in kpis:
        cards.append(f"""
        <div class="kpi-card">
          <div class="kpi-value">{value}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""")
    return "\n".join(cards)


def _table_html(df: pd.DataFrame, max_rows: int = 15) -> str:
    display_df = df.head(max_rows)
    return display_df.to_html(index=False, classes="data-table", border=0, escape=False)


def build_dashboard(
    output_path: Path,
    kpis: list[tuple[str, str, str]],
    revenue_by_category: pd.DataFrame,
    fill_rate_by_month: pd.DataFrame,
    history_network: pd.DataFrame,
    future_network: pd.DataFrame,
    abc_xyz: pd.DataFrame,
    expiry_risk: pd.DataFrame,
    model_performance: pd.DataFrame,
    ccc_summary: dict,
    working_capital_timeseries: pd.DataFrame,
    supplier_performance: pd.DataFrame,
    material_coverage: pd.DataFrame,
    excel_filename: str,
    generated_at: str,
) -> Path:
    div1 = _fig_to_div(_revenue_chart(revenue_by_category), "chart-revenue", include_js=True)
    div2 = _fig_to_div(_fill_rate_chart(fill_rate_by_month), "chart-fillrate")
    div3 = _fig_to_div(_forecast_chart(history_network, future_network), "chart-forecast")
    div4 = _fig_to_div(_abc_xyz_heatmap(abc_xyz), "chart-abcxyz")
    div5 = _fig_to_div(_category_bar_counts(abc_xyz), "chart-segcounts")
    div6 = _fig_to_div(_ccc_gauge(ccc_summary), "chart-ccc-gauge")
    div7 = _fig_to_div(_ccc_components_chart(ccc_summary), "chart-ccc-components")
    div8 = _fig_to_div(_ccc_trend_chart(working_capital_timeseries), "chart-ccc-trend")
    div9 = _fig_to_div(_supplier_ontime_chart(supplier_performance), "chart-supplier-ontime")
    div10 = _fig_to_div(_material_coverage_chart(material_coverage), "chart-material-coverage")

    expiry_display = expiry_risk[[
        "generic_name", "category", "current_network_stock", "units_expiring_30d",
        "forecasted_demand_30d", "at_risk_units_30d", "at_risk_value", "waste_risk_flag",
    ]].head(12).rename(columns={
        "generic_name": "Product", "category": "Category", "current_network_stock": "Stock on hand",
        "units_expiring_30d": "Expiring \u226430d", "forecasted_demand_30d": "Forecast demand (30d)",
        "at_risk_units_30d": "At-risk units", "at_risk_value": "At-risk value ($)",
        "waste_risk_flag": "Flag",
    })
    expiry_display["Flag"] = expiry_display["Flag"].map(lambda x: "\u26a0\ufe0f At risk" if x else "OK")

    coverage_display = material_coverage[[
        "material_name", "material_category", "current_stock", "days_of_cover", "avg_lead_time_days", "at_risk",
    ]].head(12).rename(columns={
        "material_name": "Material", "material_category": "Category", "current_stock": "Stock on hand",
        "days_of_cover": "Days of cover", "avg_lead_time_days": "Supplier lead time (d)", "at_risk": "Flag",
    })
    coverage_display["Days of cover"] = coverage_display["Days of cover"].round(1)
    coverage_display["Flag"] = coverage_display["Flag"].map(lambda x: "\u26a0\ufe0f At risk" if x else "OK")

    perf_display = model_performance.reset_index().rename(columns={"segment": "Segment"})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PharmaPulse — Demand Planning Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --ink: #4A2A20; --muted: #9A6B58; --bg: #FFF8F4; --card: #FFFFFF;
    --accent: #C0392B; --accent2: #E67E22; --border: #F3E0D6;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: var(--bg); color: var(--ink);
  }}
  header {{
    background: linear-gradient(135deg, #7B241C, #C0392B 55%, #E67E22); color: white; padding: 28px 40px;
  }}
  header h1 {{ margin: 0 0 4px 0; font-size: 26px; }}
  header p {{ margin: 0; color: #FCE7D6; font-size: 14px; }}
  .container {{ max-width: 1320px; margin: 0 auto; padding: 24px 40px 60px; }}
  .kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
  .kpi-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 22px; flex: 1 1 190px; box-shadow: 0 1px 3px rgba(192,57,43,0.06);
    border-top: 3px solid var(--accent2);
  }}
  .kpi-value {{ font-size: 26px; font-weight: 700; color: var(--accent); }}
  .kpi-label {{ font-size: 13px; font-weight: 600; margin-top: 4px; }}
  .kpi-sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .section-title {{
    font-size: 18px; font-weight: 700; margin: 34px 0 12px; color: var(--ink);
    border-left: 4px solid var(--accent2); padding-left: 10px;
  }}
  .chart-grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
  .chart-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px;
    flex: 1 1 45%; min-width: 340px;
  }}
  .chart-card.third {{ flex: 1 1 30%; min-width: 300px; }}
  .full-width {{ flex: 1 1 100%; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.data-table th {{
    background: var(--accent); color: white; padding: 8px 10px; text-align: left; position: sticky; top: 0;
  }}
  table.data-table td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); }}
  table.data-table tr:nth-child(even) {{ background: #FFF3EA; }}
  .table-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px; overflow-x: auto;
  }}
  footer {{
    max-width: 1320px; margin: 0 auto; padding: 20px 40px 40px; color: var(--muted); font-size: 12px;
  }}
  a {{ color: var(--accent); }}
  .badge {{
    display: inline-block; background: #FDEBD0; color: #7B241C; border-radius: 6px;
    padding: 2px 8px; font-size: 11px; font-weight: 600; margin-left: 8px;
  }}
</style>
</head>
<body>
<header>
  <h1>PharmaPulse — Demand Planning &amp; Inventory Intelligence <span class="badge">SYNTHETIC DEMO DATA</span></h1>
  <p>Aurora Pharmaceuticals · Network-wide report · Generated {generated_at} ·
     Full detail available in <strong>{excel_filename}</strong></p>
</header>
<div class="container">

  <div class="kpi-row">
    {_kpi_cards_html(kpis)}
  </div>

  <div class="section-title">Portfolio &amp; Service Level</div>
  <div class="chart-grid">
    <div class="chart-card">{div1}</div>
    <div class="chart-card">{div2}</div>
  </div>

  <div class="section-title">Demand Forecast (Global LightGBM Quantile Model)</div>
  <div class="chart-grid">
    <div class="chart-card full-width">{div3}</div>
  </div>

  <div class="section-title">ABC × XYZ Inventory Segmentation</div>
  <div class="chart-grid">
    <div class="chart-card">{div4}</div>
    <div class="chart-card">{div5}</div>
  </div>

  <div class="section-title">Working Capital — Cash Conversion Cycle</div>
  <div class="chart-grid">
    <div class="chart-card third">{div6}</div>
    <div class="chart-card third">{div7}</div>
    <div class="chart-card third">
      <div style="padding:14px;">
        <p style="font-size:13px; color:var(--ink); line-height:1.6;">
          <strong>CCC = DIO + DSO − DPO ≈ {ccc_summary['CCC_days']:.0f} days</strong><br>
          Inventory (finished goods + raw materials) turns in <strong>{ccc_summary['DIO_days']:.0f} days</strong>.
          Customer receivables (hospitals/pharmacies) take <strong>{ccc_summary['DSO_days']:.0f} days</strong> to collect.
          Supplier payables are extended <strong>{ccc_summary['DPO_days']:.0f} days</strong>.
          As of {ccc_summary['as_of_date']}.
        </p>
      </div>
    </div>
    <div class="chart-card full-width">{div8}</div>
  </div>

  <div class="section-title">Raw Material Supply &amp; Production Risk</div>
  <div class="chart-grid">
    <div class="chart-card">{div9}</div>
    <div class="chart-card">{div10}</div>
  </div>
  <div class="table-card" style="margin-top:20px;">
    {_table_html(coverage_display)}
  </div>

  <div class="section-title">Top Expiry / Waste-Risk Products (next 30 days)</div>
  <div class="table-card">
    {_table_html(expiry_display)}
  </div>

  <div class="section-title">Forecast Model Performance (held-out backtest)</div>
  <div class="table-card">
    {_table_html(perf_display)}
  </div>

</div>
<footer>
  PharmaPulse is a data-engineering / data-science portfolio project. All company, plant, distribution-center,
  supplier, and financial figures are synthetically generated — see <code>docs/data_dictionary.md</code> in the
  repository for full data-provenance notes. Not an official record of any real company.
</footer>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
