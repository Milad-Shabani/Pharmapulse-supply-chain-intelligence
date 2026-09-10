"""
Builds the weekly-grain modeling dataset used by the global demand
forecasting model.

Why weekly, and why one global model?
--------------------------------------
At daily grain the raw fact table has ~430K rows across 588
(product, center) series - technically forecastable series-by-series,
but that means fitting and maintaining 588 separate models, most of
which have too little signal-to-noise at the daily level to forecast
well individually. Real-world demand-planning systems at this scale
almost always use a **single global model** trained across all series
together, with the series identity (product, center, category) encoded
as categorical features - the model borrows statistical strength
across similar SKUs/centers, which is especially valuable for
lower-volume SKUs. Aggregating to weekly grain also matches how
pharma distribution planning actually operates (replenishment cycles
are weekly, not daily) and reduces noise substantially.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "units_ordered"


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demand = pd.read_parquet(raw_dir / "daily_demand.parquet")
    products = pd.read_csv(raw_dir / "products.csv")
    centers = pd.read_csv(raw_dir / "distribution_centers.csv")
    return demand, products, centers


def to_weekly(demand: pd.DataFrame) -> pd.DataFrame:
    df = demand.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    weekly = (
        df.groupby(["product_id", "center_id", "week_start"], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "weekly_units"})
    )
    return weekly


def build_feature_dataset(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand, products, centers = load_raw(raw_dir)
    weekly = to_weekly(demand)
    weekly = weekly.merge(products[["product_id", "category", "seasonality_profile"]], on="product_id")
    weekly = weekly.merge(centers[["center_id", "region", "market_size_weight"]], on="center_id")
    weekly = weekly.sort_values(["product_id", "center_id", "week_start"]).reset_index(drop=True)

    grp = weekly.groupby(["product_id", "center_id"])
    for lag in [1, 2, 3, 4, 8, 52]:
        weekly[f"lag_{lag}"] = grp["weekly_units"].shift(lag)
    for window in [4, 8, 12]:
        weekly[f"rollmean_{window}"] = grp["weekly_units"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        weekly[f"rollstd_{window}"] = grp["weekly_units"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=2).std()
        )
    weekly["diff_1"] = grp["weekly_units"].diff(1)
    weekly["diff_52"] = grp["weekly_units"].diff(52)

    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week_start"].dt.month
    weekly["year"] = weekly["week_start"].dt.year
    n_weeks_total = weekly["week_start"].nunique()
    week_index_map = {w: i for i, w in enumerate(sorted(weekly["week_start"].unique()))}
    weekly["week_index"] = weekly["week_start"].map(week_index_map)
    weekly["woy_sin"] = np.sin(2 * np.pi * weekly["week_of_year"] / 52)
    weekly["woy_cos"] = np.cos(2 * np.pi * weekly["week_of_year"] / 52)

    for cat_col in ["product_id", "center_id", "category", "region", "seasonality_profile"]:
        weekly[f"{cat_col}_code"] = weekly[cat_col].astype("category").cat.codes

    return weekly, products


FEATURE_COLUMNS = [
    "week_index", "week_of_year", "month", "woy_sin", "woy_cos",
    "market_size_weight",
    "product_id_code", "center_id_code", "category_code", "region_code", "seasonality_profile_code",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_52",
    "rollmean_4", "rollmean_8", "rollmean_12",
    "rollstd_4", "rollstd_8", "rollstd_12",
    "diff_1", "diff_52",
]
TARGET = "weekly_units"
