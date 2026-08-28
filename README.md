# Heat Intelligence — FortyGuard Hackathon'26, Track 05 (Model Designing)

An end-to-end ML system that turns raw hourly temperature data into three
operational outputs, all sharing one feature pipeline:

| Model | Question it answers | Technique |
|---|---|---|
| **Heat Forecasting Model** | What will the temperature be in the next 1–6 hours? | Gradient Boosting regression on lag/rolling/time features, recursive multi-step forecast |
| **Anomaly Detector** | Is this reading unusual given recent conditions? | IsolationForest (multivariate) + rolling z-score (statistical), flags on either firing |
| **Risk Classifier** | What heat-risk band are conditions heading into? | RandomForest, predicts the risk band *ahead of time* from trend features, not just from today's reading |

## Why this design

- **One feature pipeline, three model heads.** `src/features.py` is the only
  place lag windows, rolling stats, and the heat-index/risk-band thresholds
  are defined. All three models consume it, so they can't silently drift out
  of sync with each other — a single source of truth you can defend in front
  of judges.
- **Works with zero setup.** `src/data_client.py` tries FortyGuard first when
  configured, then falls back to Open-Meteo and synthetic data, so the whole thing
  runs offline with no API key for development, then upgrades to real data
  with one flag once your FortyGuard credits are live.
- **Every design choice is commented with its trade-off**, not just its
  behavior — e.g. why recursive forecasting compounds error, why heat index
  falls back to a fixed humidity assumption, why anomalies need two signals
  instead of one. Read the docstrings before you present; they're written as
  answers to the questions a judge will actually ask.

## Project structure

```
fortyguard-heat-intelligence/
├── src/
│   ├── data_client.py      # data sources: FortyGuard / Open-Meteo / synthetic
│   ├── features.py         # shared feature engineering + heat index + risk labels
│   ├── forecasting.py       # Heat Forecasting Model
│   ├── anomaly.py           # Anomaly Detector
│   ├── risk_classifier.py   # Risk Classifier
│   ├── pipeline.py          # wires the above into one object
│   └── api.py                # Flask API: GET /heat-intelligence
├── run_demo.py               # one-command CLI demo
├── requirements.txt
└── outputs/                  # trained models + latest snapshot land here
```

## Where to run this

**For building and demoing (recommended):** your own laptop, locally.
Nothing here needs a GPU or heavy resources — training all three models on
30 days of hourly data takes a few seconds on a normal laptop. Python
3.10+ is all you need.

**For a live link judges can open without you present:** deploy the Flask
app to a free host once it's working locally — Render.com or Railway.app
both auto-detect a Flask app from `requirements.txt` and a start command
(`cd src && python api.py`, or better, `gunicorn` — see note below) and
give you a public URL in a few minutes, no credit card for Render's free
tier. PythonAnywhere is another simple option built specifically for
small Python/Flask apps. Any of these is enough for a hackathon demo link;
you don't need AWS/GCP scale infrastructure for this.

Two small things to do before deploying (not needed for local use):
1. Use `gunicorn app:app --bind 0.0.0.0:$PORT` as the start command — Flask's own
  development server isn't meant to be exposed publicly. A ready-to-use
  `render.yaml` is included for Render.
2. Set `FORTYGUARD_API_KEY` as an environment variable in the host's
   dashboard instead of hardcoding it in `data_client.py`, so the key isn't
   sitting in your public GitHub repo if you push one.

### Deploy on Render

1. Push this folder to a GitHub repository (never commit the API key).
2. In Render, choose **New > Blueprint** and select the repository.
3. Render reads `render.yaml`, installs `requirements.txt`, and starts the
  dashboard with Gunicorn.
4. In the service's Environment settings, add `FORTYGUARD_API_KEY` if the
  FortyGuard endpoint is available. The app remains usable with Open-Meteo
  fallback when that variable is omitted or the endpoint is unavailable.
5. Open the generated `https://...onrender.com` URL and append `/health` to
  confirm the deployment returns `{"status":"ok"}`.

## Quickstart

```bash
pip install -r requirements.txt
python run_demo.py                       # console demo — synthetic data, no key needed
python run_demo.py --source open_meteo    # console demo — real historical data, no key needed
python run_demo.py --source fortyguard    # live FortyGuard data + history fallback
```

**Dashboard (the actual UI):**

```bash
cd src && python api.py
```

Then open **http://localhost:5000** in a browser — that's the dashboard:
current reading, a thermal risk gauge, a 6-hour forecast chart, and a live
anomaly log, auto-refreshing every 60 seconds. `GET /api/heat-intelligence`
still returns the raw JSON if you want to hit it directly:

```bash
curl "http://localhost:5000/api/heat-intelligence?location=Islamabad&lat=33.6844&lon=73.0479&forecast_hours=6"
```

## Plugging in the real FortyGuard Temperature API

Set `FORTYGUARD_API_KEY` as an environment variable before running the app:

```python
# =====================================================================
# PASTE YOUR FORTYGUARD API KEY HERE
# =====================================================================
FORTYGUARD_API_KEY = ""
# =====================================================================
```

For local testing, export your key in the shell rather than committing it.
That's the only setup required — `get_temperature_series(source="auto")`
detects the key automatically and switches from Open-Meteo/synthetic data
to a real FortyGuard reading blended with recent history (see the
`_fetch_fortyguard_snapshot` docstring in that file for exactly how, and
why it's a blend rather than 100% FortyGuard data).

If your hackathon account's response shape differs from what's assumed here
(`temperature_f`, optionally wrapped in `{"data": {...}}`), that's also all
inside `_fetch_fortyguard_snapshot` — nothing else in the project needs to
change.

## Honest limitations (say these out loud when presenting — it reads as
rigor, not weakness)

- Recursive multi-step forecasting compounds error hour over hour; good
  enough for a 6h operational window, not for day-ahead forecasting.
- Heat index uses a fixed 40% relative-humidity assumption when the data
  source doesn't provide humidity — swap in real RH the moment it's
  available (FortyGuard's richer tiers may expose it).
- Risk-band thresholds are a defensible starting point (WHO/NWS-style
  bands adapted to °C), not a validated medical standard — call this out
  rather than presenting it as clinically authoritative.
- IsolationForest is fit on ~30 days of history in the demo; more history
  = a better sense of "normal" and fewer false positives in production.

## Extending toward other tracks

The Flask API in `src/api.py` returns forecast + risk + anomalies as one
JSON payload — a Dashboard track team could render it directly, a Maps
track team could call it per-grid-cell, and an Agents track team could wrap
`/heat-intelligence` as a tool call. Built to be a building block, not just
a standalone entry.
