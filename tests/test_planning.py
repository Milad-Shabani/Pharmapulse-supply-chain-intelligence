import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pharmapulse.data_generation.products import products_dataframe
from pharmapulse.data_generation.centers_and_plants import centers_dataframe
from pharmapulse.data_generation.demand_curve import generate_daily_demand
from pharmapulse.planning.abc_xyz import compute_abc_xyz
from pharmapulse.planning.safety_stock import compute_safety_stock


def test_abc_classes_are_valid_and_a_is_minority_of_lines():
    products = products_dataframe()
    demand = generate_daily_demand(seed=1)
    result = compute_abc_xyz(demand, products)
    assert set(result["abc_class"].unique()) <= {"A", "B", "C"}
    assert set(result["xyz_class"].astype(str).unique()) <= {"X", "Y", "Z"}
    # Pareto principle: A-class should be a minority of (product,center) lines
    # but should not be empty.
    counts = result["abc_class"].value_counts()
    assert 0 < counts.get("A", 0) < len(result)


def test_abc_revenue_shares_sum_to_one():
    products = products_dataframe()
    demand = generate_daily_demand(seed=1)
    result = compute_abc_xyz(demand, products)
    assert abs(result["revenue_share"].sum() - 1.0) < 1e-6


def test_safety_stock_non_negative_and_ordered():
    centers = centers_dataframe()
    forecast = pd.DataFrame({
        "product_id": ["SKU-001"] * 2,
        "center_id": centers["center_id"].iloc[:2].tolist(),
        "week_start": pd.Timestamp("2025-01-06"),
        "forecast_p50": [100.0, 50.0],
        "forecast_p95": [180.0, 90.0],
    })
    result = compute_safety_stock(forecast, centers)
    assert (result["safety_stock_units"] >= 0).all()
    assert (result["reorder_point_units"] >= result["expected_demand_over_lead_time"]).all()
    assert (result["order_up_to_units"] >= result["reorder_point_units"]).all()


def test_safety_stock_scales_with_lead_time():
    """A longer lead time should never produce a smaller reorder point,
    all else equal."""
    centers = pd.DataFrame({
        "center_id": ["SHORT", "LONG"],
        "lead_time_days_from_plant": [2, 10],
    })
    forecast = pd.DataFrame({
        "product_id": ["SKU-001", "SKU-001"],
        "center_id": ["SHORT", "LONG"],
        "week_start": pd.Timestamp("2025-01-06"),
        "forecast_p50": [100.0, 100.0],
        "forecast_p95": [150.0, 150.0],
    })
    result = compute_safety_stock(forecast, centers).set_index("center_id")
    assert result.loc["LONG", "reorder_point_units"] > result.loc["SHORT", "reorder_point_units"]
