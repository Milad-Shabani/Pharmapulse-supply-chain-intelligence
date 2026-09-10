"""
Cash Conversion Cycle (CCC) and its three components - Days Inventory
Outstanding (DIO), Days Sales Outstanding (DSO), and Days Payable
Outstanding (DPO):

    CCC = DIO + DSO - DPO

DIO is computed directly from the *actual* simulated inventory value
(finished goods + raw materials) and cost of goods sold - it is a
genuine output of the operational simulation, not an assumption.

DSO and DPO are driven by a mechanistic accounts-receivable /
accounts-payable ledger: AR grows with each day's revenue and shrinks
when the revenue booked `dso_target_days` earlier is collected; AP
works the same way against COGS with a `dpo_target_days` payment lag.
This produces genuinely fluctuating (not just hard-coded) DSO/DPO
series that hover near their target collection/payment terms -
Aurora's receivables terms with hospitals/pharmacies are deliberately
set long (order-to-cash in healthcare distribution is notoriously
slow) while payables terms are comparatively tight, which is what
pushes network CCC to the ~60-day level referenced throughout the
dashboard.

All dollar figures are trailing-window (default 60 days) averages,
matching how CCC is conventionally reported and matching the
dashboard's headline "60-day Cash Conversion Cycle" framing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mechanistic_ledger(flow: pd.Series, term_days: int) -> pd.Series:
    """
    Simulates a receivable/payable balance: each day's flow (revenue
    or COGS) increases the balance; the flow booked `term_days`
    earlier is realized (collected/paid) and decreases it.

    Pre-history (before day 0) is assumed to run at day-0's flow rate
    (steady state), implemented by padding the front of the series
    with `term_days` copies of the first value, running the exact
    same lagged recurrence over the padded series, then dropping the
    padding - this avoids double-counting day 0 against the seed
    balance and lets DSO/DPO settle cleanly at `term_days` once the
    initial transient clears.
    """
    n = len(flow)
    values = flow.to_numpy()
    if n == 0:
        return pd.Series([], index=flow.index)

    pad = np.full(term_days, values[0])
    padded = np.concatenate([pad, values])
    balance_padded = np.zeros(len(padded))
    running = 0.0
    for t in range(len(padded)):
        running += padded[t]
        realized_idx = t - term_days
        if realized_idx >= 0:
            running -= padded[realized_idx]
        balance_padded[t] = max(running, 0)
    balance = balance_padded[term_days:]
    return pd.Series(balance, index=flow.index)


def build_working_capital_timeseries(
    finished_goods_daily: pd.DataFrame,  # date, inv_value_fg, cogs, revenue
    raw_material_daily: pd.DataFrame,    # date, inv_value_rm
    dso_target_days: int = 75,
    dpo_target_days: int = 30,
    window: int = 60,
) -> pd.DataFrame:
    df = finished_goods_daily.merge(raw_material_daily, on="date", how="left").sort_values("date")
    df["inv_value_rm"] = df["inv_value_rm"].ffill().fillna(0)
    df["total_inventory_value"] = df["inv_value_fg"] + df["inv_value_rm"]

    df["accounts_receivable"] = _mechanistic_ledger(df.set_index("date")["revenue"], dso_target_days).to_numpy()
    df["accounts_payable"] = _mechanistic_ledger(df.set_index("date")["cogs"], dpo_target_days).to_numpy()

    df["avg_inventory_w"] = df["total_inventory_value"].rolling(window, min_periods=10).mean()
    df["avg_cogs_w"] = df["cogs"].rolling(window, min_periods=10).mean()
    df["avg_revenue_w"] = df["revenue"].rolling(window, min_periods=10).mean()
    df["avg_ar_w"] = df["accounts_receivable"].rolling(window, min_periods=10).mean()
    df["avg_ap_w"] = df["accounts_payable"].rolling(window, min_periods=10).mean()

    df["DIO"] = df["avg_inventory_w"] / df["avg_cogs_w"]
    df["DSO"] = df["avg_ar_w"] / df["avg_revenue_w"]
    df["DPO"] = df["avg_ap_w"] / df["avg_cogs_w"]
    df["CCC"] = df["DIO"] + df["DSO"] - df["DPO"]

    return df


def summarize_latest(df: pd.DataFrame) -> dict:
    latest = df.dropna(subset=["CCC"]).iloc[-1]
    return {
        "DIO_days": round(latest["DIO"], 1),
        "DSO_days": round(latest["DSO"], 1),
        "DPO_days": round(latest["DPO"], 1),
        "CCC_days": round(latest["CCC"], 1),
        "as_of_date": latest["date"],
    }
