"""
api.py
=======
Serves two things:

  GET /                        -> the dashboard UI (templates/index.html)
  GET /api/heat-intelligence   -> the same data as JSON, for the UI's own
                                   fetch() calls or for other tracks
                                   (Agents/Dashboards/Maps) to consume

Run:  python api.py
Then open http://localhost:5000 in a browser.
"""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

try:
    from .pipeline import HeatIntelligencePipeline
except ImportError:
    from pipeline import HeatIntelligencePipeline

app = Flask(__name__)

# Cache one fitted pipeline per (lat, lon) so repeated requests don't retrain
# from scratch — fine for a hackathon demo; swap for a proper model registry
# / scheduled retrain job for anything longer-lived.
_pipeline_cache: dict[tuple[float, float], HeatIntelligencePipeline] = {}


def _get_pipeline(location: str, lat: float, lon: float) -> HeatIntelligencePipeline:
    key = (round(lat, 3), round(lon, 3))
    if key not in _pipeline_cache:
        _pipeline_cache[key] = HeatIntelligencePipeline(location=location, lat=lat, lon=lon).fit()
    return _pipeline_cache[key]


@app.get("/")
def dashboard():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/heat-intelligence")
def heat_intelligence():
    location = request.args.get("location", "Islamabad, PK")
    lat = float(request.args.get("lat", 33.6844))
    lon = float(request.args.get("lon", 73.0479))
    forecast_hours = int(request.args.get("forecast_hours", 6))

    try:
        pipeline = _get_pipeline(location, lat, lon)
        result = pipeline.run(forecast_hours=forecast_hours)
        return jsonify(result)
    except Exception as exc:  # keep the demo resilient during judging
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
