"""
Generates the core large fact table: daily unit demand for every
(product, distribution center) pair across a 2-year study window
(2023-01-01 -> 2024-12-31, 730 days).

With 42 SKUs x 14 centers x 730 days, this is ~412,000 rows - enough
to be a genuine "large data" exercise (requires chunked/vectorized
generation, benefits from Parquet storage, and makes naive per-row
Python loops impractically slow, which is itself part of why the
project uses numpy-vectorized generation and a global ML model rather
than one model per series).

Design
------
Expected daily demand for (product p, center c, day d) is built as:

    base(p) * market_weight(c) * seasonal(p, d) * trend(p, d) * promo(p, d)

then sampled from a Negative Binomial distribution (captures the
over-dispersion real pharmacy order data shows - burstier than a
Poisson process - while still being non-negative integer counts).

Seasonality profiles (`seasonality_profile` on each product) map to
smooth cosine-based curves anchored to real epidemiological seasonal
patterns: respiratory-illness and antibiotic demand peaks in winter,
allergy medication peaks in spring/summer, vitamin demand spikes in
January, and chronic-disease medication (cardiovascular, diabetes,
endocrine) stays essentially flat with only a slow secular uptrend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STUDY_START = pd.Timestamp("2023-01-01")
STUDY_END = pd.Timestamp("2024-12-31")


def _day_of_year_angle(dates: pd.DatetimeIndex) -> np.ndarray:
    doy = dates.dayofyear.to_numpy(dtype=float)
    return 2 * np.pi * doy / 365.25


def _seasonal_factor(profile: str, dates: pd.DatetimeIndex) -> np.ndarray:
    theta = _day_of_year_angle(dates)
    if profile in ("flat", "flat_chronic"):
        return np.ones(len(dates))
    if profile == "winter_bump":
        return 1.0 + 0.28 * np.cos(theta)
    if profile == "strong_winter":
        return 1.0 + 0.55 * np.cos(theta)
    if profile == "mild_winter":
        return 1.0 + 0.15 * np.cos(theta)
    if profile == "spring_summer":
        return 1.0 + 0.45 * np.cos(theta - np.pi)
    if profile == "strong_summer":
        return 1.0 + 0.55 * np.cos(theta - np.pi)
    if profile == "new_year_bump":
        doy = dates.dayofyear.to_numpy(dtype=float)
        bump = 0.9 * np.exp(-((doy - 10) ** 2) / (2 * 18 ** 2))
        bump += 0.9 * np.exp(-((doy - 375) ** 2) / (2 * 18 ** 2))
        return 1.0 + bump
    return np.ones(len(dates))


def _trend_factor(profile: str, day_index: np.ndarray) -> np.ndarray:
    annual_growth = 0.10 if profile == "flat_chronic" else 0.06
    return 1.0 + annual_growth * (day_index / 365.25)


def _promo_multiplier(rng: np.random.Generator, category: str, n_days: int) -> np.ndarray:
    promo_prone = {"Analgesics", "Vitamins", "Respiratory", "Gastrointestinal"}
    mult = np.ones(n_days)
    if category not in promo_prone:
        return mult
    n_events = rng.integers(4, 7)
    for _ in range(n_events):
        start = rng.integers(0, n_days - 14)
        length = rng.integers(5, 12)
        uplift = rng.uniform(1.3, 1.9)
        mult[start : start + length] *= uplift
    return mult


def generate_daily_demand(seed: int = 2024) -> pd.DataFrame:
    from pharmapulse.data_generation.products import products_dataframe
    from pharmapulse.data_generation.centers_and_plants import centers_dataframe

    rng = np.random.default_rng(seed)
    products = products_dataframe()
    centers = centers_dataframe()
    dates = pd.date_range(STUDY_START, STUDY_END, freq="D")
    n_days = len(dates)
    day_index = np.arange(n_days)

    frames = []
    for prod in products.itertuples(index=False):
        seasonal = _seasonal_factor(prod.seasonality_profile, dates)
        trend = _trend_factor(prod.seasonality_profile, day_index)
        promo = _promo_multiplier(rng, prod.category, n_days)
        weekday_factor = np.where(dates.weekday.isin([5, 6]), 0.55, 1.0)

        product_day_factor = seasonal * trend * promo * weekday_factor

        for center in centers.itertuples(index=False):
            expected = prod.base_daily_units_per_center * center.market_size_weight * product_day_factor
            expected = np.clip(expected, 0.05, None)

            dispersion = 6.0
            gamma_shape = dispersion
            gamma_scale = expected / dispersion
            lam = rng.gamma(gamma_shape, gamma_scale)
            units = rng.poisson(lam)

            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "product_id": prod.product_id,
                        "center_id": center.center_id,
                        "units_ordered": units,
                    }
                )
            )

    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


if __name__ == "__main__":
    import time

    t0 = time.time()
    df = generate_daily_demand()
    print(f"Generated {len(df):,} rows in {time.time() - t0:.1f}s")
    print(df.head())
