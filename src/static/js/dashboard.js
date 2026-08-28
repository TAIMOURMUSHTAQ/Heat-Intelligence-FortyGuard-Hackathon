/* ===========================================================
   Heat Intelligence dashboard
   Fetches /api/heat-intelligence and renders the instrument
   panel: current reading + risk gauge, 6h forecast chart,
   anomaly log. No external chart library — the forecast line
   is drawn by hand into the inline SVG so it stays visually
   consistent with the rest of the instrument.
   =========================================================== */

const RISK_ORDER = ["low", "moderate", "high", "very_high", "extreme"];
const RISK_POSITION_PCT = { low: 10, moderate: 31, high: 52, very_high: 74, extreme: 92 };
const RISK_COLOR_VAR = {
  low: "--scale-low",
  moderate: "--scale-moderate",
  high: "--scale-high",
  very_high: "--scale-veryhigh",
  extreme: "--scale-extreme",
};
const RISK_DISPLAY = {
  low: "Low", moderate: "Moderate", high: "High",
  very_high: "Very High", extreme: "Extreme",
};

const $ = (id) => document.getElementById(id);
let refreshTimer = null;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmtTime(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function fmtHour(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleTimeString(undefined, { hour: "numeric" });
}

async function fetchIntelligence() {
  const res = await fetch("/api/heat-intelligence");
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

function renderMeta(data) {
  $("meta-location").textContent = data.location || "—";
  $("meta-asof").textContent = fmtTime(data.as_of);
  $("meta-source").textContent = (data.source || "model").toUpperCase();
}

function renderHero(data) {
  $("hero-temp").textContent = data.current_temperature_c.toFixed(1);

  const level = data.current_risk_level;
  const colorVar = RISK_COLOR_VAR[level] || "--text-muted";
  const color = cssVar(colorVar);

  $("risk-label").textContent = RISK_DISPLAY[level] || level;
  $("risk-dot").style.background = color;
  $("risk-badge").style.borderColor = color;
  $("risk-confidence").textContent = data.current_risk_confidence
    ? `${Math.round(data.current_risk_confidence * 100)}% confidence`
    : "";

  const pct = RISK_POSITION_PCT[level] ?? 50;
  const marker = $("gauge-marker");
  marker.style.left = `${pct}%`;
  $("gauge-marker-value").textContent = `${data.current_temperature_c.toFixed(1)}°C`;
}

function renderForecastChart(forecast) {
  const svg = $("forecast-chart");
  svg.innerHTML = "";
  if (!forecast || forecast.length === 0) return;

  const W = 640, H = 220, padX = 34, padY = 28;
  const temps = forecast.map((f) => f.forecast_temperature_c);
  const min = Math.min(...temps) - 0.6;
  const max = Math.max(...temps) + 0.6;
  const span = Math.max(max - min, 0.5);

  const xFor = (i) => padX + (i / (forecast.length - 1 || 1)) * (W - padX * 2);
  const yFor = (t) => H - padY - ((t - min) / span) * (H - padY * 2);

  const ns = "http://www.w3.org/2000/svg";
  const accent = cssVar("--accent");

  // gridlines
  for (let g = 0; g <= 3; g++) {
    const y = padY + (g / 3) * (H - padY * 2);
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", padX); line.setAttribute("x2", W - padX);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("stroke", cssVar("--border"));
    line.setAttribute("stroke-width", "1");
    svg.appendChild(line);

    const label = document.createElementNS(ns, "text");
    const val = max - (g / 3) * span;
    label.textContent = `${val.toFixed(1)}°`;
    label.setAttribute("x", 4); label.setAttribute("y", y + 4);
    label.setAttribute("fill", cssVar("--text-faint"));
    label.setAttribute("font-size", "10");
    label.setAttribute("font-family", "IBM Plex Mono, monospace");
    svg.appendChild(label);
  }

  // area fill
  let areaPath = `M ${xFor(0)} ${H - padY} `;
  forecast.forEach((f, i) => { areaPath += `L ${xFor(i)} ${yFor(f.forecast_temperature_c)} `; });
  areaPath += `L ${xFor(forecast.length - 1)} ${H - padY} Z`;

  const defs = document.createElementNS(ns, "defs");
  const grad = document.createElementNS(ns, "linearGradient");
  grad.setAttribute("id", "areaGrad");
  grad.setAttribute("x1", "0"); grad.setAttribute("y1", "0");
  grad.setAttribute("x2", "0"); grad.setAttribute("y2", "1");
  grad.innerHTML = `<stop offset="0%" stop-color="${accent}" stop-opacity="0.35"/>
                     <stop offset="100%" stop-color="${accent}" stop-opacity="0"/>`;
  defs.appendChild(grad);
  svg.appendChild(defs);

  const area = document.createElementNS(ns, "path");
  area.setAttribute("d", areaPath);
  area.setAttribute("fill", "url(#areaGrad)");
  svg.appendChild(area);

  // line
  let linePath = "";
  forecast.forEach((f, i) => {
    linePath += `${i === 0 ? "M" : "L"} ${xFor(i)} ${yFor(f.forecast_temperature_c)} `;
  });
  const line = document.createElementNS(ns, "path");
  line.setAttribute("d", linePath);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", accent);
  line.setAttribute("stroke-width", "2.5");
  line.setAttribute("stroke-linecap", "round");
  line.setAttribute("stroke-linejoin", "round");
  svg.appendChild(line);

  // points + hour labels
  forecast.forEach((f, i) => {
    const cx = xFor(i), cy = yFor(f.forecast_temperature_c);
    const dot = document.createElementNS(ns, "circle");
    dot.setAttribute("cx", cx); dot.setAttribute("cy", cy); dot.setAttribute("r", "3.5");
    dot.setAttribute("fill", cssVar("--bg-panel"));
    dot.setAttribute("stroke", accent);
    dot.setAttribute("stroke-width", "2");
    svg.appendChild(dot);

    const hourLabel = document.createElementNS(ns, "text");
    hourLabel.textContent = fmtHour(f.timestamp);
    hourLabel.setAttribute("x", cx);
    hourLabel.setAttribute("y", H - 6);
    hourLabel.setAttribute("text-anchor", "middle");
    hourLabel.setAttribute("fill", cssVar("--text-faint"));
    hourLabel.setAttribute("font-size", "10");
    hourLabel.setAttribute("font-family", "IBM Plex Mono, monospace");
    svg.appendChild(hourLabel);
  });

  $("forecast-range").textContent =
    `${forecast[0].forecast_temperature_c.toFixed(1)}° → ${forecast[forecast.length - 1].forecast_temperature_c.toFixed(1)}°C`;
  $("forecast-legend").innerHTML =
    `<span>+1h</span><span>+${forecast.length}h</span>`;
}

function renderAnomalies(data) {
  const list = $("anomaly-list");
  list.innerHTML = "";
  const anomalies = data.recent_anomalies || [];

  $("anomaly-rate").textContent = data.anomaly_summary
    ? `${data.anomaly_summary.n_anomalies} / ${data.anomaly_summary.n_points} pts · ${data.anomaly_summary.anomaly_rate_pct}%`
    : "—";

  if (anomalies.length === 0) {
    $("anomaly-empty").hidden = false;
    return;
  }
  $("anomaly-empty").hidden = true;

  [...anomalies].reverse().forEach((a) => {
    const li = document.createElement("li");
    li.className = "anomaly-item";
    li.innerHTML = `
      <span class="anomaly-flag"></span>
      <span class="anomaly-time">${fmtTime(a.timestamp)}</span>
      <span class="anomaly-temp">${a.temperature_c.toFixed(1)}°C</span>
      <span class="anomaly-reason">${a.anomaly_reason.replace(/_/g, " ")}</span>
    `;
    list.appendChild(li);
  });
}

function showLoading() {
  $("state-loading").hidden = false;
  $("state-error").hidden = true;
  $("app-content").hidden = true;
}

function showError(message) {
  $("state-loading").hidden = true;
  $("state-error").hidden = false;
  $("app-content").hidden = true;
  $("error-detail").textContent = message;
}

function showContent() {
  $("state-loading").hidden = true;
  $("state-error").hidden = true;
  $("app-content").hidden = false;
}

async function load({ silent = false } = {}) {
  if (!silent) showLoading();
  try {
    const data = await fetchIntelligence();
    renderMeta(data);
    renderHero(data);
    renderForecastChart(data.forecast);
    renderAnomalies(data);
    showContent();
  } catch (err) {
    if (!silent || $("app-content").hidden) {
      showError(err.message || "Unknown error");
    }
  }
}

function scheduleAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(() => load({ silent: true }), 60000);
}

$("refresh-btn").addEventListener("click", () => {
  $("refresh-btn").classList.add("spinning");
  load({ silent: true }).finally(() => {
    setTimeout(() => $("refresh-btn").classList.remove("spinning"), 700);
  });
});
$("retry-btn").addEventListener("click", () => load());

load();
scheduleAutoRefresh();
