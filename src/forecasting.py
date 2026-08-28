"""
forecasting.py
===============
Heat Forecasting Model.

Frames forecasting as supervised regression on lag/rolling/time features
(rather than a classical ARIMA/Prophet time-series model) so the exact same
feature table (features.py) and evaluation harness feed all three models in
this project — one shared feature pipeline, three separate model heads.

GradientBoostingRegressor is deliberately simple and fast to justify on a
whiteboard, trains in seconds on hackathon-scale hourly data, and gives
feature_importances_ for free, which makes the "why" behind a forecast easy
to explain during judging.

Multi-step ahead forecasting uses a recursive strategy: predict h+1, feed
that prediction back in as if it were observed, predict h+2, etc. This is a
known limitation (errors compound) — documented explicitly rather than
hidden, since a judge asking "how do you forecast 6 hours out from a
1-hour-ahead model" is a fair question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from features import build_feature_table, add_time_features, add_lag_and_rolling_features, LAGS, ROLL_WINDOWS

FEATURE_COLS = (
    [f"temp_lag_{l}h" for l in LAGS]
    + [f"roll_mean_{w}h" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}h" for w in ROLL_WINDOWS]
    + ["rate_of_change_1h", "rate_of_change_3h", "hour_sin", "hour_cos", "day_of_week"]
)
TARGET_COL = "temperature_c"


class HeatForecaster:
    def __init__(self, **model_kwargs):
        defaults = dict(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=42)
        defaults.update(model_kwargs)
        self.model = GradientBoostingRegressor(**defaults)
        self.is_fitted = False

    def fit(self, feature_df: pd.DataFrame):
        X, y = feature_df[FEATURE_COLS], feature_df[TARGET_COL]
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def evaluate(self, feature_df: pd.DataFrame) -> dict:
        X, y = feature_df[FEATURE_COLS], feature_df[TARGET_COL]
        preds = self.model.predict(X)
        return {
            "mae_c": round(mean_absolute_error(y, preds), 3),
            "rmse_c": round(root_mean_squared_error(y, preds), 3),
            "n": len(y),
        }

    def predict_next(self, feature_df: pd.DataFrame) -> float:
        """One-step-ahead forecast from the most recent feature row."""
        last_row = feature_df.iloc[[-1]][FEATURE_COLS]
        return float(self.model.predict(last_row)[0])

    def forecast_horizon(self, raw_df: pd.DataFrame, hours_ahead: int = 6) -> pd.DataFrame:
        """Recursive multi-step forecast. Returns a DataFrame of
        (timestamp, forecast_temperature_c) for hours_ahead future hours.
        """
        history = raw_df[["timestamp", "temperature_c"]].copy().sort_values("timestamp")
        results = []
        for step in range(1, hours_ahead + 1):
            feats = add_time_features(history)
            feats = add_lag_and_rolling_features(feats).dropna()
            next_temp = self.predict_next(feats)
            next_ts = history["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
            results.append({"timestamp": next_ts, "forecast_temperature_c": round(next_temp, 2),
                             "step_ahead_h": step})
            history = pd.concat(
                [history, pd.DataFrame([{"timestamp": next_ts, "temperature_c": next_temp}])],
                ignore_index=True,
            )
        return pd.DataFrame(results)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)


if __name__ == "__main__":
    from data_client import get_temperature_series

    raw = get_temperature_series(days_back=30)
    feats = build_feature_table(raw)

    split = int(len(feats) * 0.85)
    train, test = feats.iloc[:split], feats.iloc[split:]

    model = HeatForecaster().fit(train)
    print("Test set performance:", model.evaluate(test))
    print("\nTop features:\n", model.feature_importance().head(6))

    horizon = model.forecast_horizon(raw, hours_ahead=6)
    print("\n6h-ahead forecast:\n", horizon)
