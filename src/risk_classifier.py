"""
risk_classifier.py
====================
Heat Risk Classifier.

Labels (low / moderate / high / very_high / extreme) come from
features.label_risk_level(), a fixed heat-index threshold table — that part
is a lookup, not ML, and is deliberately kept as simple as possible so it's
auditable (a city/insurer/logistics user needs to be able to see exactly
where a boundary sits).

The ML piece is predicting that SAME risk band ahead of time, from features
that don't require already knowing the future heat index — i.e. from the
current/recent temperature trajectory alone. That's the part actually worth
a model: "given how conditions have been trending, what risk band are we
heading into?", which is the useful operational question (staff a cooling
center, delay outdoor work, etc.) versus just reading a thermometer.

RandomForestClassifier is used for the same reasons as the forecaster:
handles the mixed feature scales fine with no tuning, gives
feature_importances_, and predict_proba() gives a confidence per band
instead of a bare label, which is more useful for a risk dashboard.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

try:
    from .forecasting import FEATURE_COLS
except ImportError:
    from forecasting import FEATURE_COLS

TARGET_COL = "risk_level"
RISK_ORDER = ["low", "moderate", "high", "very_high", "extreme"]
TARGET_HORIZON_H = 1


class HeatRiskClassifier:
    def __init__(self, **model_kwargs):
        defaults = dict(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
        defaults.update(model_kwargs)
        self.model = RandomForestClassifier(**defaults)
        self.is_fitted = False

    def fit(self, feature_df: pd.DataFrame):
        labeled = feature_df.copy()
        labeled["future_risk_level"] = labeled[TARGET_COL].shift(-TARGET_HORIZON_H)
        labeled = labeled.dropna(subset=["future_risk_level"])
        X = labeled[FEATURE_COLS]
        y = labeled["future_risk_level"].astype(str)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def evaluate(self, feature_df: pd.DataFrame) -> str:
        labeled = feature_df.copy()
        labeled["future_risk_level"] = labeled[TARGET_COL].shift(-TARGET_HORIZON_H)
        labeled = labeled.dropna(subset=["future_risk_level"])
        X = labeled[FEATURE_COLS]
        y = labeled["future_risk_level"].astype(str)
        preds = self.model.predict(X)
        labels = [l for l in RISK_ORDER if l in set(y) | set(preds)]
        return classification_report(y, preds, labels=labels, zero_division=0)

    def predict_with_confidence(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        X = feature_df[FEATURE_COLS]
        probs = self.model.predict_proba(X)
        classes = self.model.classes_
        preds = self.model.predict(X)
        confidence = probs.max(axis=1)
        return pd.DataFrame({
            "timestamp": feature_df["timestamp"].values,
            "predicted_risk_level": preds,
            "confidence": confidence.round(3),
        })

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)


if __name__ == "__main__":
    from data_client import get_temperature_series
    from features import build_feature_table

    raw = get_temperature_series(days_back=30)
    feats = build_feature_table(raw)

    split = int(len(feats) * 0.85)
    train, test = feats.iloc[:split], feats.iloc[split:]

    clf = HeatRiskClassifier().fit(train)
    print(clf.evaluate(test))
    print("\nTop features:\n", clf.feature_importance().head(6))

    preds = clf.predict_with_confidence(test)
    print("\nSample predictions:\n", preds.tail(8).to_string(index=False))
