import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pharmapulse.models import evaluate


def test_wape_zero_error():
    y = np.array([10.0, 20.0, 30.0])
    assert evaluate.wape(y, y) == 0


def test_wape_known_value():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 110.0])
    # |{-10, 10}| sum = 20, over sum(|y_true|) = 200 -> 10%
    assert evaluate.wape(y_true, y_pred) == 10.0


def test_bias_pct_sign():
    y_true = np.array([100.0, 100.0])
    over_pred = np.array([110.0, 110.0])
    under_pred = np.array([90.0, 90.0])
    assert evaluate.bias_pct(y_true, over_pred) > 0
    assert evaluate.bias_pct(y_true, under_pred) < 0


def test_summarize_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    result = evaluate.summarize(y_true, y_pred, "seg")
    assert set(result.keys()) == {"segment", "MAE", "RMSE", "WAPE_%", "MAPE_%", "Bias_%", "R2", "n"}


def test_weekly_aggregation_preserves_total_units():
    from pharmapulse.features.build_features import to_weekly

    daily = pd.DataFrame({
        "date": pd.date_range("2023-01-02", periods=14, freq="D").strftime("%Y-%m-%d"),  # starts on a Monday
        "product_id": "SKU-001",
        "center_id": "DC-01",
        "units_ordered": range(14),
    })
    weekly = to_weekly(daily)
    assert weekly["weekly_units"].sum() == daily["units_ordered"].sum()
    assert len(weekly) == 2  # 14 days = 2 full weeks
