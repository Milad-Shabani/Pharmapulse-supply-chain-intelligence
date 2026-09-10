"""
A single **global** LightGBM model trained across all 588
(product, center) series simultaneously, with series identity encoded
as categorical features (`product_id`, `center_id`, `category`,
`region`, `seasonality_profile`) - the standard architecture for
demand forecasting at retail/distribution scale (this is essentially
the approach used in the M5 forecasting competition and in production
systems at large retailers/distributors).

Two models are trained on the SAME feature set but different LightGBM
`quantile` objectives:

  * tau = 0.50 (median forecast) - the number to show a planner as
    "expected demand".
  * tau = 0.95 - directly usable as a **demand-side safety buffer**:
    ordering to the P95 forecast rather than the P50 forecast is a
    quantile-regression-native alternative to the classical
    "mean + z*sigma" safety-stock formula, and it captures each
    series' actual (possibly skewed, non-Normal) demand distribution
    rather than assuming Normality.

Both are combined with supplier lead time in
`pharmapulse.planning.safety_stock` to produce reorder-point
recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from pharmapulse.features.build_features import FEATURE_COLUMNS, TARGET

CATEGORICAL_FEATURES = [
    "product_id_code", "center_id_code", "category_code", "region_code", "seasonality_profile_code",
]

BASE_PARAMS = dict(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.04,
    min_child_samples=25,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
)


@dataclass
class QuantileModels:
    p50: lgb.LGBMRegressor
    p95: lgb.LGBMRegressor
    feature_importance_p50: pd.Series


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    required = ["lag_1", "lag_2", "lag_3", "lag_4", "rollmean_4"]
    return df.dropna(subset=required)


def train_quantile_models(train_df: pd.DataFrame) -> QuantileModels:
    clean = _prep(train_df)
    X = clean[FEATURE_COLUMNS]
    y = clean[TARGET]

    p50_model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **BASE_PARAMS)
    p50_model.fit(X, y, categorical_feature=CATEGORICAL_FEATURES)

    p95_model = lgb.LGBMRegressor(objective="quantile", alpha=0.95, **BASE_PARAMS)
    p95_model.fit(X, y, categorical_feature=CATEGORICAL_FEATURES)

    importance = pd.Series(p50_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    return QuantileModels(p50=p50_model, p95=p95_model, feature_importance_p50=importance)


def predict(models: QuantileModels, feature_df: pd.DataFrame) -> pd.DataFrame:
    X = feature_df[FEATURE_COLUMNS]
    p50 = np.clip(models.p50.predict(X), 0, None)
    p95 = np.clip(models.p95.predict(X), 0, None)
    p95 = np.maximum(p95, p50)
    return pd.DataFrame({"forecast_p50": p50, "forecast_p95": p95}, index=feature_df.index)


def recursive_forecast(
    models: QuantileModels,
    weekly_history: pd.DataFrame,
    n_future_weeks: int,
    static_lookup: pd.DataFrame,
) -> pd.DataFrame:
    working = weekly_history.copy()
    last_week = working["week_start"].max()
    future_weeks = [last_week + pd.Timedelta(weeks=i) for i in range(1, n_future_weeks + 1)]

    series_keys = working[["product_id", "center_id"]].drop_duplicates()
    all_new_rows = []

    for week in future_weeks:
        new_rows = series_keys.copy()
        new_rows["week_start"] = week
        new_rows[TARGET] = np.nan
        working = pd.concat([working, new_rows], ignore_index=True)
        working = working.merge(
            static_lookup, on=["product_id", "center_id"], how="left", suffixes=("", "_dup")
        )
        working = working[[c for c in working.columns if not c.endswith("_dup")]]

        featured = _recompute_features(working)
        this_week_mask = featured["week_start"] == week
        preds = predict(models, featured.loc[this_week_mask])
        working.loc[this_week_mask, TARGET] = preds["forecast_p50"].to_numpy()

        snapshot = featured.loc[this_week_mask, ["product_id", "center_id", "week_start"]].copy()
        snapshot["forecast_p50"] = preds["forecast_p50"].to_numpy()
        snapshot["forecast_p95"] = preds["forecast_p95"].to_numpy()
        all_new_rows.append(snapshot)

    return pd.concat(all_new_rows, ignore_index=True)


def _recompute_features(working: pd.DataFrame) -> pd.DataFrame:
    df = working.sort_values(["product_id", "center_id", "week_start"]).reset_index(drop=True)
    grp = df.groupby(["product_id", "center_id"])
    for lag in [1, 2, 3, 4, 8, 52]:
        df[f"lag_{lag}"] = grp[TARGET].shift(lag)
    for window in [4, 8, 12]:
        df[f"rollmean_{window}"] = grp[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rollstd_{window}"] = grp[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=2).std()
        )
    df["diff_1"] = grp[TARGET].diff(1)
    df["diff_52"] = grp[TARGET].diff(52)

    df["week_of_year"] = pd.to_datetime(df["week_start"]).dt.isocalendar().week.astype(int)
    df["month"] = pd.to_datetime(df["week_start"]).dt.month
    week_index_map = {w: i for i, w in enumerate(sorted(df["week_start"].unique()))}
    df["week_index"] = df["week_start"].map(week_index_map)
    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    for cat_col in ["product_id", "center_id", "category", "region", "seasonality_profile"]:
        df[f"{cat_col}_code"] = df[cat_col].astype("category").cat.codes
    return df
