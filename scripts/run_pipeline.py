"""
End-to-end PharmaPulse pipeline:

1. Load raw synthetic data (run generate_sample_data.py first).
2. Build the weekly, global-model feature set.
3. Backtest the LightGBM P50/P95 quantile model on a held-out 12-week
   window, with accuracy metrics reported overall and by category.
4. Refit on full history and produce a 12-week forward forecast.
5. Compute ABC/XYZ segmentation, quantile-based safety stock &
   reorder-point recommendations, and expiry/waste risk.
6. Compute the Cash Conversion Cycle (DIO/DSO/DPO) and raw-material
   supply-chain risk metrics (supplier on-time delivery, days of
   material cover vs. lead time).
7. Render both the Excel planning workbook and the HTML dashboard from
   the same computed tables.

Usage:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore")

from pharmapulse.features.build_features import TARGET, build_feature_dataset  # noqa: E402
from pharmapulse.models import evaluate, global_forecast_model as gfm  # noqa: E402
from pharmapulse.planning.abc_xyz import compute_abc_xyz  # noqa: E402
from pharmapulse.planning.safety_stock import compute_safety_stock  # noqa: E402
from pharmapulse.planning.expiry_risk import compute_expiry_risk  # noqa: E402
from pharmapulse.planning.working_capital import (  # noqa: E402
    build_working_capital_timeseries, summarize_latest,
)
from pharmapulse.planning.supply_risk import (  # noqa: E402
    compute_supplier_performance, compute_material_coverage,
)
from pharmapulse.reporting.excel_report import build_excel_report  # noqa: E402
from pharmapulse.reporting.html_dashboard import build_dashboard  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
N_TEST_WEEKS = 12
N_FUTURE_WEEKS = 12
DSO_TARGET_DAYS = 75
DPO_TARGET_DAYS = 30
CCC_WINDOW_DAYS = 60


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(RAW_DIR / "products.csv")
    centers = pd.read_csv(RAW_DIR / "distribution_centers.csv")
    daily_demand = pd.read_parquet(RAW_DIR / "daily_demand.parquet")
    batches = pd.read_csv(RAW_DIR / "production_batches.csv")
    inventory = pd.read_parquet(RAW_DIR / "inventory_ledger.parquet")
    materials = pd.read_csv(RAW_DIR / "raw_materials.csv")
    suppliers = pd.read_csv(RAW_DIR / "suppliers.csv")
    purchase_orders = pd.read_csv(RAW_DIR / "purchase_orders.csv")
    rm_inventory = pd.read_parquet(RAW_DIR / "raw_material_inventory.parquet")

    # ------------------------------------------------------------------
    # 1. Features + backtest
    # ------------------------------------------------------------------
    print("[1/8] Building weekly feature dataset...")
    weekly, _ = build_feature_dataset(RAW_DIR)
    weekly.to_parquet(PROCESSED_DIR / "weekly_features.parquet", index=False)

    weeks = sorted(weekly["week_start"].unique())
    test_weeks = weeks[-N_TEST_WEEKS:]
    train_df = weekly[~weekly["week_start"].isin(test_weeks)]
    test_df = weekly[weekly["week_start"].isin(test_weeks)].copy()

    print(f"[2/8] Training global LightGBM quantile models on {len(train_df):,} rows, "
          f"backtesting on the final {N_TEST_WEEKS} weeks ({len(test_df):,} rows)...")
    backtest_models = gfm.train_quantile_models(train_df)
    test_preds = gfm.predict(backtest_models, test_df)
    test_df["forecast_p50"] = test_preds["forecast_p50"].to_numpy()
    test_df["forecast_p95"] = test_preds["forecast_p95"].to_numpy()

    overall_perf = evaluate.summarize(test_df[TARGET].to_numpy(), test_df["forecast_p50"].to_numpy(), "Network overall")
    coverage = float((test_df[TARGET] <= test_df["forecast_p95"]).mean() * 100)
    perf_rows = [overall_perf]
    for cat, g in test_df.groupby("category"):
        perf_rows.append(evaluate.summarize(g[TARGET].to_numpy(), g["forecast_p50"].to_numpy(), cat))
    model_performance = evaluate.metrics_table(perf_rows)
    model_performance["P95_coverage_%"] = np.nan
    model_performance.loc["Network overall", "P95_coverage_%"] = round(coverage, 1)
    model_performance.to_csv(REPORTS_DIR / "model_performance.csv")
    print(model_performance)

    # ------------------------------------------------------------------
    # 2. Refit on full history, forecast forward
    # ------------------------------------------------------------------
    print(f"\n[3/8] Refitting on full history and forecasting {N_FUTURE_WEEKS} weeks forward...")
    full_models = gfm.train_quantile_models(weekly)
    static_lookup = weekly[
        ["product_id", "center_id", "category", "region", "seasonality_profile", "market_size_weight"]
    ].drop_duplicates()
    future = gfm.recursive_forecast(
        full_models, weekly[["product_id", "center_id", "week_start", TARGET]], N_FUTURE_WEEKS, static_lookup
    )
    future = future.merge(products[["product_id", "category"]], on="product_id")
    future.to_parquet(PROCESSED_DIR / "future_forecast_12w.parquet", index=False)

    # ------------------------------------------------------------------
    # 3. Planning layer: ABC/XYZ, safety stock, expiry risk
    # ------------------------------------------------------------------
    print("[4/8] Computing ABC/XYZ segmentation, safety stock, and expiry risk...")
    abc_xyz = compute_abc_xyz(daily_demand, products)
    abc_xyz = abc_xyz.merge(products[["product_id", "generic_name", "category"]], on="product_id")

    inventory_plan = compute_safety_stock(future, centers)
    inventory_plan = inventory_plan.merge(products[["product_id", "generic_name"]], on="product_id")
    inventory_plan = inventory_plan.merge(
        abc_xyz[["product_id", "center_id", "segment"]], on=["product_id", "center_id"], how="left"
    )
    next_week = inventory_plan["week_start"].min()
    inventory_plan_next = inventory_plan[inventory_plan["week_start"] == next_week][
        ["product_id", "generic_name", "category", "center_id", "segment", "week_start",
         "daily_forecast_p50", "daily_forecast_p95", "reorder_point_units", "order_up_to_units"]
    ].sort_values("reorder_point_units", ascending=False)
    inventory_plan_next.to_csv(REPORTS_DIR / "inventory_plan.csv", index=False)

    as_of_date = pd.to_datetime(daily_demand["date"]).max()
    current_stock_by_product = inventory[
        pd.to_datetime(inventory["date"]) == as_of_date
    ].groupby("product_id")["closing_stock"].sum()
    forecast_by_product_week = future.groupby(["product_id", "week_start"], as_index=False)["forecast_p50"].sum()
    expiry_risk = compute_expiry_risk(batches, current_stock_by_product, forecast_by_product_week, as_of_date, products)
    expiry_risk.to_csv(REPORTS_DIR / "expiry_risk.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Working capital: Cash Conversion Cycle
    # ------------------------------------------------------------------
    print("[5/8] Computing the Cash Conversion Cycle (DIO / DSO / DPO)...")
    fg = inventory.merge(products[["product_id", "unit_cost", "unit_price"]], on="product_id")
    fg["inv_value_fg"] = fg["closing_stock"] * fg["unit_cost"]
    fg["cogs"] = fg["units_sold"] * fg["unit_cost"]
    fg["revenue"] = fg["units_sold"] * fg["unit_price"]
    fg_daily = fg.groupby("date", as_index=False).agg(
        inv_value_fg=("inv_value_fg", "sum"), cogs=("cogs", "sum"), revenue=("revenue", "sum")
    )

    rm = rm_inventory.merge(materials[["material_id", "unit_cost"]], on="material_id")
    rm["inv_value_rm"] = rm["closing_stock"] * rm["unit_cost"]
    rm_daily = rm.groupby("date", as_index=False)["inv_value_rm"].sum()

    working_capital = build_working_capital_timeseries(
        fg_daily, rm_daily, dso_target_days=DSO_TARGET_DAYS, dpo_target_days=DPO_TARGET_DAYS, window=CCC_WINDOW_DAYS
    )
    working_capital.to_csv(REPORTS_DIR / "working_capital.csv", index=False)
    ccc_summary = summarize_latest(working_capital)
    print(f"  -> DIO={ccc_summary['DIO_days']}d  DSO={ccc_summary['DSO_days']}d  "
          f"DPO={ccc_summary['DPO_days']}d  CCC={ccc_summary['CCC_days']}d (as of {ccc_summary['as_of_date']})")

    # ------------------------------------------------------------------
    # 5. Raw-material supply & production risk
    # ------------------------------------------------------------------
    print("[6/8] Computing supplier performance and raw-material coverage risk...")
    supplier_performance = compute_supplier_performance(purchase_orders, suppliers)
    supplier_performance.to_csv(REPORTS_DIR / "supplier_performance.csv", index=False)

    material_coverage = compute_material_coverage(
        rm_inventory, materials, as_of_date=as_of_date.strftime("%Y-%m-%d"), trailing_window_days=CCC_WINDOW_DAYS
    )
    material_coverage.to_csv(REPORTS_DIR / "material_coverage.csv", index=False)
    n_at_risk = int(material_coverage["at_risk"].sum())
    print(f"  -> {n_at_risk} of {len(material_coverage)} materials below safe days-of-cover")

    # ------------------------------------------------------------------
    # 6. Executive summary tables
    # ------------------------------------------------------------------
    print("[7/8] Assembling executive-summary tables...")
    merged = daily_demand.merge(products[["product_id", "unit_price", "category"]], on="product_id")
    merged["revenue"] = merged["units_ordered"] * merged["unit_price"]
    total_revenue = merged["revenue"].sum()
    revenue_by_category = merged.groupby("category", as_index=False)["revenue"].sum().rename(
        columns={"revenue": "total_revenue"}
    ).sort_values("total_revenue", ascending=False)

    inv_dated = inventory.copy()
    inv_dated["date"] = pd.to_datetime(inv_dated["date"])
    inv_dated["month"] = inv_dated["date"].dt.to_period("M").astype(str)
    monthly = inv_dated.groupby("month", as_index=False).agg(
        demand=("units_demanded", "sum"), stockout=("stockout_units", "sum")
    )
    monthly["fill_rate"] = 1 - monthly["stockout"] / monthly["demand"]

    total_units = int(merged["units_ordered"].sum())
    overall_fill_rate = 1 - inventory["stockout_units"].sum() / inventory["units_demanded"].sum()
    at_risk_value_total = expiry_risk["at_risk_value"].sum() if not expiry_risk.empty else 0.0
    overall_on_time = float(purchase_orders["on_time"].mean())

    kpis = [
        ("Network Revenue (2Y)", f"${total_revenue/1e6:,.1f}M", "Across 42 SKUs · 14 distribution centers"),
        ("Units Sold (2Y)", f"{total_units/1e6:,.2f}M", "Daily demand fact table, 429,828 rows"),
        ("Network Fill Rate", f"{overall_fill_rate:.1%}", "Current (s,S) policy — degrading YoY, see report"),
        ("Cash Conversion Cycle", f"{ccc_summary['CCC_days']:.0f} days", f"DIO {ccc_summary['DIO_days']:.0f}d + DSO {ccc_summary['DSO_days']:.0f}d − DPO {ccc_summary['DPO_days']:.0f}d"),
        ("Forecast Accuracy (WAPE)", f"{overall_perf['WAPE_%']:.1f}%", f"Held-out {N_TEST_WEEKS}-week backtest, global LightGBM model"),
        ("Supplier On-Time Rate", f"{overall_on_time:.1%}", f"{len(purchase_orders):,} purchase orders across {len(suppliers)} suppliers"),
        ("Expiry Risk (30d)", f"${at_risk_value_total:,.0f}", "Forecast-adjusted waste exposure"),
    ]

    exec_summary_cells = {
        "Network Revenue (2Y)": f"${total_revenue/1e6:,.1f}M",
        "Units Sold (2Y)": f"{total_units/1e6:,.2f}M",
        "Network Fill Rate": f"{overall_fill_rate:.1%}",
        "Cash Conversion Cycle": f"{ccc_summary['CCC_days']:.0f} days",
        "Forecast WAPE": f"{overall_perf['WAPE_%']:.1f}%",
        "Supplier On-Time Rate": f"{overall_on_time:.1%}",
    }

    forecast_by_category_week = future.groupby(["category", "week_start"], as_index=False).agg(
        forecast_p50=("forecast_p50", "sum"), forecast_p95=("forecast_p95", "sum")
    ).round(0)

    history_network = weekly.groupby("week_start", as_index=False)[TARGET].sum()
    history_network = history_network[history_network["week_start"] >= history_network["week_start"].max() - pd.Timedelta(weeks=26)]
    future_network = future.groupby("week_start", as_index=False).agg(
        forecast_p50=("forecast_p50", "sum"), forecast_p95=("forecast_p95", "sum")
    )

    # ------------------------------------------------------------------
    # 7. Excel report + HTML dashboard
    # ------------------------------------------------------------------
    print("[8/8] Writing Excel workbook and HTML dashboard...")
    excel_path = REPORTS_DIR / "PharmaPulse_Planning_Report.xlsx"
    build_excel_report(
        excel_path,
        exec_summary=exec_summary_cells,
        revenue_by_category=revenue_by_category,
        fill_rate_by_month=monthly[["month", "fill_rate"]],
        forecast_by_category_week=forecast_by_category_week,
        inventory_plan=inventory_plan_next,
        abc_xyz=abc_xyz[["product_id", "generic_name", "category", "center_id", "total_revenue",
                          "demand_cv", "abc_class", "xyz_class", "segment"]],
        expiry_risk=expiry_risk,
        model_performance=model_performance,
        ccc_summary=ccc_summary,
        working_capital_timeseries=working_capital,
        supplier_performance=supplier_performance,
        material_coverage=material_coverage,
    )

    dashboard_path = REPORTS_DIR / "pharmapulse_dashboard.html"
    build_dashboard(
        dashboard_path,
        kpis=kpis,
        revenue_by_category=revenue_by_category,
        fill_rate_by_month=monthly[["month", "fill_rate"]],
        history_network=history_network,
        future_network=future_network,
        abc_xyz=abc_xyz,
        expiry_risk=expiry_risk,
        model_performance=model_performance,
        ccc_summary=ccc_summary,
        working_capital_timeseries=working_capital,
        supplier_performance=supplier_performance,
        material_coverage=material_coverage,
        excel_filename=excel_path.name,
        generated_at=datetime.now().strftime("%Y-%m-%d"),
    )

    print(f"\nDone.\n  Excel report : {excel_path}\n  HTML dashboard: {dashboard_path}")


if __name__ == "__main__":
    main()
