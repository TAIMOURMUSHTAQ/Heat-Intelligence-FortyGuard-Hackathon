"""
run_demo.py
============
One-command, no-setup demo of the full pipeline. Run this to sanity-check
everything before a live demo, or to generate the numbers/plots you'll put
in your submission.

    python run_demo.py

Uses synthetic data by default (works with zero network access / no API
key). Point --source at open_meteo or fortyguard once you have real access;
see src/data_client.py.
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import HeatIntelligencePipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="FortyGuard Heat Intelligence — end-to-end demo")
    parser.add_argument("--location", default="Islamabad, PK")
    parser.add_argument("--lat", type=float, default=33.6844)
    parser.add_argument("--lon", type=float, default=73.0479)
    parser.add_argument("--days-back", type=int, default=30)
    parser.add_argument("--forecast-hours", type=int, default=6)
    parser.add_argument("--source", default="auto", choices=["auto", "fortyguard", "open_meteo", "synthetic"])
    args = parser.parse_args()

    print(f"Fetching + training on {args.days_back} days of hourly data for {args.location}...")
    pipeline = HeatIntelligencePipeline(
        location=args.location, lat=args.lat, lon=args.lon,
        days_back=args.days_back, source=args.source,
    ).fit()

    print(f"\nForecast evaluation (chronological holdout):  {pipeline.forecaster.evaluate(pipeline._test_feats)}")
    print(f"Anomaly summary:                {pipeline.detector.summary(pipeline.detector.detect(pipeline._last_feats))}")

    result = pipeline.run(forecast_hours=args.forecast_hours)
    print("\n=== Heat Intelligence Snapshot ===")
    print(json.dumps(result, indent=2, default=str))

    os.makedirs("outputs", exist_ok=True)
    pipeline.save(path_prefix="outputs/model")
    with open("outputs/latest_snapshot.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved trained models + snapshot to outputs/")


if __name__ == "__main__":
    main()
