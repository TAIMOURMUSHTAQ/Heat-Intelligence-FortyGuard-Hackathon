"""
data_client.py
================
Data access layer for the project. Three sources, in priority order:

1. FortyGuard Temperature API (real hackathon data) — used once you paste
   your API key below. Pulls a real live reading from FortyGuard's
   heat-intelligence endpoint and blends it with recent history from
   Open-Meteo so the lag/rolling features the models need still have
   something to work with (FortyGuard's free-tier point endpoint returns
   a live snapshot, not an hourly backfill — see the docstring on
   `_fetch_fortyguard_snapshot` below for the honest version of this).
2. Open-Meteo (free, no key, real weather data) — used as a live fallback so
   the whole pipeline can be developed and demoed before FortyGuard credits
   are issued, or if the API is rate-limited during the event.
3. Synthetic generator — deterministic fake-but-realistic hourly series, used
   for unit tests / offline demos with zero network dependency.

Everything downstream (features.py, forecasting.py, anomaly.py,
risk_classifier.py) only depends on the pandas.DataFrame contract returned
by `get_temperature_series()`:

    timestamp (datetime64), location (str), temperature_c (float)

Swap sources freely — the rest of the pipeline does not change.
"""
from __future__ import annotations

import os
import time
import math
import datetime as dt

import numpy as np
import pandas as pd
import requests

FORTYGUARD_BASE_URL = "https://api.fortyguard.com/v1"

# =====================================================================
# PASTE YOUR FORTYGUARD API KEY HERE
# =====================================================================
# Set FORTYGUARD_API_KEY in the environment when live data is available.
# Leave this empty to use Open-Meteo / synthetic data without credentials.
FORTYGUARD_API_KEY = ""
# =====================================================================


class FortyGuardClient:
    """Thin wrapper around FortyGuard's Temperature API.

    Two call shapes are implemented, matching the two patterns FortyGuard
    documents publicly:

    - `heat_intelligence()`  — synchronous point lookup (location in,
      temperature + risk_level out). Good fit for this project: one call
      per location per refresh.
    - `submit_heatmap_job()` / `poll_job()` — async polygon-area job for
      their /heatmap endpoint (submit -> activity_id -> poll until done).
      Not used by this project's pipeline today, but included since a
      Dashboards/Maps-track teammate may want area coverage rather than a
      single point.

    NOTE: confirm exact field names against the docs you were handed at
    kickoff — this was built from FortyGuard's public product pages, and
    hackathon accounts occasionally get slightly different response
    shapes. This class isolates that risk to one file.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or FORTYGUARD_API_KEY or os.environ.get("FORTYGUARD_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No FortyGuard API key configured. Paste it into FORTYGUARD_API_KEY "
                "near the top of src/data_client.py."
            )
        self.headers = {"api-key": self.api_key, "Content-Type": "application/json"}

    def heat_intelligence(self, location: str, lat: float, lon: float) -> dict:
        """Synchronous point lookup: current temperature + risk band for a place."""
        payload = {"location": location, "latitude": lat, "longitude": lon}
        resp = requests.post(f"{FORTYGUARD_BASE_URL}/heat-intelligence", headers=self.headers,
                              json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def submit_heatmap_job(self, polygon_aoi: dict, start_date: str, start_time: str,
                            granularity: int = 100) -> str:
        payload = {
            "polygon_aoi": polygon_aoi,
            "date_time": {"start_date": start_date, "start_time": start_time, "filter_type": 1},
            "granularity": granularity,
        }
        resp = requests.post(f"{FORTYGUARD_BASE_URL}/heatmap", headers=self.headers,
                              json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["data"]["activity_id"]

    def poll_job(self, activity_id: str, poll_every_s: float = 2.0, timeout_s: float = 120.0) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = requests.get(f"{FORTYGUARD_BASE_URL}/heatmap/{activity_id}",
                                 headers=self.headers, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            status = body.get("data", {}).get("status")
            if status == "completed":
                return body["data"]
            if status == "failed":
                raise RuntimeError(f"FortyGuard job {activity_id} failed: {body}")
            time.sleep(poll_every_s)
        raise TimeoutError(f"FortyGuard job {activity_id} did not complete in {timeout_s}s")


def _fetch_open_meteo(lat: float, lon: float, days_back: int = 30) -> pd.DataFrame:
    """Real hourly 2m temperature history from Open-Meteo (no key required)."""
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        "&hourly=temperature_2m&timezone=auto"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    js = resp.json()["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(js["time"]),
        "temperature_c": js["temperature_2m"],
    })
    return df


def _fetch_fortyguard_snapshot(location: str, lat: float, lon: float, days_back: int) -> pd.DataFrame:
    """Hybrid fetch: a real, live reading from FortyGuard for the current
    hour, stitched onto Open-Meteo history for the lag/rolling window the
    models need.

    Why hybrid rather than 100% FortyGuard: the models need an hourly
    backfill (lags up to 24h, rolling windows up to 24h) to produce a
    forecast/risk read at all, and the free-tier point endpoint FortyGuard
    documents publicly returns one live reading per call, not a historical
    series. If your hackathon account has a historical/backfill endpoint,
    swap the Open-Meteo call below for repeated calls to that endpoint —
    the rest of the pipeline doesn't care where the hourly rows came from.
    """
    history = _fetch_open_meteo(lat, lon, days_back)

    client = FortyGuardClient()
    live = client.heat_intelligence(location, lat, lon)
    live_data = live.get("data", live)  # some endpoints wrap in {"data": {...}}
    live_temp_f = live_data.get("temperature_f")
    if live_temp_f is None:
        raise RuntimeError(f"Unexpected FortyGuard response shape: {live}")
    live_temp_c = (live_temp_f - 32) * 5 / 9
    live_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now("UTC").floor("h"),
        "temperature_c": live_temp_c,
    }])

    combined = pd.concat([history, live_row], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return combined.reset_index(drop=True)


def _synthetic_series(hours: int = 24 * 30, seed: int = 42, base_c: float = 33.0,
                       inject_anomalies: bool = True) -> pd.DataFrame:
    """Deterministic synthetic hourly temperature series for a hot-climate city
    (calibrated loosely to Islamabad summer highs), including a daily cycle,
    a slow seasonal drift, noise, and a few injected heat-spike anomalies so
    the anomaly detector has something real to find during offline demos.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.now("UTC").floor("h") - pd.Timedelta(hours=hours)
    timestamps = pd.date_range(start=start, periods=hours, freq="h")

    hour_of_day = timestamps.hour.values
    day_index = np.arange(hours) / 24.0

    daily_cycle = 6.0 * np.sin((hour_of_day - 9) / 24.0 * 2 * math.pi)
    seasonal_drift = 0.03 * day_index
    noise = rng.normal(0, 0.6, size=hours)

    temps = base_c + daily_cycle + seasonal_drift + noise

    if inject_anomalies and hours > 96:
        anomaly_idx = rng.choice(np.arange(48, hours - 48), size=max(1, hours // 200), replace=False)
        for idx in anomaly_idx:
            spike = rng.uniform(4.5, 8.0)
            width = rng.integers(2, 6)
            temps[idx: idx + width] += spike

    return pd.DataFrame({"timestamp": timestamps, "temperature_c": temps})


def get_temperature_series(location: str = "Islamabad, PK", lat: float = 33.6844,
                            lon: float = 73.0479, days_back: int = 30,
                            source: str = "auto") -> pd.DataFrame:
    """Unified entry point used by the rest of the pipeline.

    source: "fortyguard" | "open_meteo" | "synthetic" | "auto"
        auto uses FortyGuard when FORTYGUARD_API_KEY is set above, falls
        back to Open-Meteo, and finally to the synthetic generator if both
        fail (e.g. no network access).
    """
    def tag(df: pd.DataFrame, src: str) -> pd.DataFrame:
        df = df.copy()
        df["location"] = location
        df["source"] = src
        return df.sort_values("timestamp").reset_index(drop=True)

    has_key = bool(FORTYGUARD_API_KEY or os.environ.get("FORTYGUARD_API_KEY"))

    if source == "fortyguard" or (source == "auto" and has_key):
        try:
            return tag(_fetch_fortyguard_snapshot(location, lat, lon, days_back), "fortyguard+open_meteo")
        except Exception:
            if source == "fortyguard":
                raise
            # auto mode: fall through to open_meteo below

    if source in ("open_meteo", "auto"):
        try:
            return tag(_fetch_open_meteo(lat, lon, days_back), "open_meteo")
        except Exception:
            if source == "open_meteo":
                raise

    return tag(_synthetic_series(hours=days_back * 24), "synthetic")


if __name__ == "__main__":
    df = get_temperature_series()
    print(df.head())
    print(f"\n{len(df)} rows from source={df['source'].iloc[0]}")
