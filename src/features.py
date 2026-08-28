"""
features.py
============
Turns a raw (timestamp, temperature_c) series into a feature table that the
three models share:

  - lag features        -> forecasting model's inputs
  - rolling mean/std     -> anomaly detector's inputs
  - heat_index_c + label -> risk classifier's target

Keeping this in one module means "what counts as a feature" is defined once,
so the three models can't silently drift out of sync with each other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12, 24]           # hours
ROLL_WINDOWS = [3, 6, 24]             # hours


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in LAGS:
        df[f"temp_lag_{lag}h"] = df["temperature_c"].shift(lag)
    for window in ROLL_WINDOWS:
        df[f"roll_mean_{window}h"] = df["temperature_c"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}h"] = df["temperature_c"].shift(1).rolling(window).std()
    df["rate_of_change_1h"] = df["temperature_c"].diff(1)
    df["rate_of_change_3h"] = df["temperature_c"].diff(3)
    return df


def compute_heat_index_c(temp_c: pd.Series, relative_humidity_pct: float = 40.0) -> pd.Series:
    """NOAA/Rothfusz heat-index regression, approximated with a fixed RH when
    humidity isn't available from the data source (FortyGuard's 2m
    temperature layer doesn't always ship humidity in the free tier).
    Returns heat index in Celsius. Good enough for relative risk banding —
    NOT a substitute for a medical/meteorological heat-index reading.
    """
    temp_f = temp_c * 9 / 5 + 32
    rh = relative_humidity_pct
    hi_f = (
        -42.379 + 2.04901523 * temp_f + 10.14333127 * rh
        - 0.22475541 * temp_f * rh - 0.00683783 * temp_f ** 2
        - 0.05481717 * rh ** 2 + 0.00122874 * temp_f ** 2 * rh
        + 0.00085282 * temp_f * rh ** 2 - 0.00000199 * temp_f ** 2 * rh ** 2
    )
    # Rothfusz regression is only valid above ~80F; below that, heat index ~= air temp.
    hi_f = np.where(temp_f < 80, temp_f, hi_f)
    return (hi_f - 32) * 5 / 9


def label_risk_level(heat_index_c: pd.Series) -> pd.Series:
    """WHO/NWS-style heat risk bands, adapted to Celsius thresholds.
    These thresholds are a defensible starting point for the demo — swap in
    FortyGuard's own risk_level bands if the API exposes them directly.
    """
    bins = [-np.inf, 27, 32, 39, 46, np.inf]
    labels = ["low", "moderate", "high", "very_high", "extreme"]
    return pd.cut(heat_index_c, bins=bins, labels=labels)


def build_feature_table(df: pd.DataFrame, relative_humidity_pct: float = 40.0) -> pd.DataFrame:
    """Full pipeline: raw series -> model-ready feature table (with NaNs from
    the lag/rolling warm-up window dropped)."""
    out = df.sort_values("timestamp").reset_index(drop=True)
    out = add_time_features(out)
    out = add_lag_and_rolling_features(out)
    out["heat_index_c"] = compute_heat_index_c(out["temperature_c"], relative_humidity_pct)
    out["risk_level"] = label_risk_level(out["heat_index_c"])
    return out.dropna().reset_index(drop=True)


if __name__ == "__main__":
    from data_client import get_temperature_series
    raw = get_temperature_series(days_back=30)
    feats = build_feature_table(raw)
    print(feats[["timestamp", "temperature_c", "heat_index_c", "risk_level"]].tail(10))
    print(f"\n{len(feats)} feature rows, {feats.shape[1]} columns")
