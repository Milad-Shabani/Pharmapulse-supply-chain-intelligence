import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pharmapulse.planning.working_capital import build_working_capital_timeseries, summarize_latest
from pharmapulse.planning.supply_risk import compute_supplier_performance, compute_material_coverage


def _synthetic_flows(n_days: int = 400, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D").strftime("%Y-%m-%d")
    revenue = rng.uniform(900, 1100, n_days)
    cogs = revenue * 0.4
    inv_value_fg = rng.uniform(4000, 5000, n_days)
    fg_daily = pd.DataFrame({"date": dates, "inv_value_fg": inv_value_fg, "cogs": cogs, "revenue": revenue})
    rm_daily = pd.DataFrame({"date": dates, "inv_value_rm": rng.uniform(2000, 3000, n_days)})
    return fg_daily, rm_daily


def test_ccc_settles_near_target_terms():
    fg_daily, rm_daily = _synthetic_flows()
    wc = build_working_capital_timeseries(fg_daily, rm_daily, dso_target_days=75, dpo_target_days=30, window=60)
    tail = wc.dropna(subset=["CCC"]).tail(100)
    # DSO/DPO should settle close to their target collection/payment terms
    assert abs(tail["DSO"].mean() - 75) < 8
    assert abs(tail["DPO"].mean() - 30) < 8


def test_ccc_equals_components_identity():
    fg_daily, rm_daily = _synthetic_flows()
    wc = build_working_capital_timeseries(fg_daily, rm_daily)
    tail = wc.dropna(subset=["CCC"])
    computed = tail["DIO"] + tail["DSO"] - tail["DPO"]
    pd.testing.assert_series_equal(computed, tail["CCC"], check_names=False)


def test_summarize_latest_keys():
    fg_daily, rm_daily = _synthetic_flows()
    wc = build_working_capital_timeseries(fg_daily, rm_daily)
    summary = summarize_latest(wc)
    assert set(summary.keys()) == {"DIO_days", "DSO_days", "DPO_days", "CCC_days", "as_of_date"}


def test_supplier_performance_on_time_rate_between_0_and_1():
    purchase_orders = pd.DataFrame({
        "po_id": [f"PO-{i}" for i in range(6)],
        "supplier_id": ["SUP-01", "SUP-01", "SUP-01", "SUP-02", "SUP-02", "SUP-02"],
        "on_time": [True, True, False, True, True, True],
        "delay_days": [0, 0, 5, 0, 0, 0],
        "quantity_ordered": [100, 100, 100, 50, 50, 50],
        "unit_cost": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    })
    suppliers = pd.DataFrame({
        "supplier_id": ["SUP-01", "SUP-02"],
        "supplier_name": ["A", "B"],
        "region": ["Domestic", "Overseas"],
        "category": ["Packaging", "API"],
    })
    perf = compute_supplier_performance(purchase_orders, suppliers)
    assert ((perf["on_time_rate"] >= 0) & (perf["on_time_rate"] <= 1)).all()
    row = perf.set_index("supplier_id").loc["SUP-01"]
    assert abs(row["on_time_rate"] - (2 / 3)) < 1e-9


def test_material_coverage_flags_low_stock_as_at_risk():
    dates = pd.date_range("2023-01-01", periods=10).strftime("%Y-%m-%d")
    rm_inventory = pd.DataFrame({
        "date": list(dates) * 1,
        "material_id": ["MAT-1"] * 10,
        "closing_stock": [5] * 10,       # very low stock
        "qty_consumed": [10] * 10,       # high consumption -> ~0.5 days of cover
    })
    materials = pd.DataFrame({
        "material_id": ["MAT-1"],
        "material_name": ["Test Material"],
        "material_category": ["API"],
        "supplier_id": ["SUP-01"],
        "avg_lead_time_days": [30],
    })
    coverage = compute_material_coverage(rm_inventory, materials, as_of_date=dates[-1], trailing_window_days=10)
    assert coverage.iloc[0]["at_risk"] == True  # noqa: E712 - clarity over strict style here
