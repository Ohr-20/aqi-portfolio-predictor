from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs
import gzip
import html
import json
import math


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "model_assets"

with open(ASSET_DIR / "model_data.json", "r", encoding="utf-8") as f:
    MODEL_DATA = json.load(f)

with gzip.open(ASSET_DIR / "xgb_classifier.json.gz", "rt", encoding="utf-8") as f:
    CLASSIFIER_MODEL = json.load(f)

with gzip.open(ASSET_DIR / "xgb_regressor.json.gz", "rt", encoding="utf-8") as f:
    REGRESSOR_MODEL = json.load(f)

TARGET_CLASSES = MODEL_DATA["target_classes"]
CATEGORICAL_FEATURES = MODEL_DATA["categorical_features"]
NUMERIC_FEATURES = MODEL_DATA["numeric_features"]
CLASSIFIER_CATEGORIES = MODEL_DATA["classifier_categories"]
REGRESSOR_CATEGORIES = MODEL_DATA["regressor_categories"]


def softmax(values):
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def get_booster_parts(model_json):
    learner = model_json["learner"]
    model = learner["gradient_booster"]["model"]
    trees = model["trees"]
    tree_info = model["tree_info"]
    base_scores = json.loads(learner["learner_model_param"]["base_score"])
    return trees, tree_info, base_scores


CLASS_TREES, CLASS_TREE_INFO, CLASS_BASE_SCORES = get_booster_parts(CLASSIFIER_MODEL)
REG_TREES, REG_TREE_INFO, REG_BASE_SCORES = get_booster_parts(REGRESSOR_MODEL)


def tree_predict(tree, features):
    node = 0
    while tree["left_children"][node] != -1:
        split_index = tree["split_indices"][node]
        split_value = tree["split_conditions"][node]
        if split_index not in features:
            node = tree["left_children"][node] if tree["default_left"][node] else tree["right_children"][node]
            continue
        feature_value = features[split_index]
        if feature_value < split_value:
            node = tree["left_children"][node]
        else:
            node = tree["right_children"][node]
    return tree["base_weights"][node]


def transform_features(row, categories_by_feature):
    transformed = {}
    output_index = 0
    for feature, categories in zip(CATEGORICAL_FEATURES, categories_by_feature):
        value = str(row[feature])
        for category in categories:
            if value == str(category):
                transformed[output_index] = 1.0
            output_index += 1
    for feature in NUMERIC_FEATURES:
        value = float(row[feature])
        if value != 0.0:
            transformed[output_index] = value
        output_index += 1
    return transformed


def predict_xgb_classifier(features):
    scores = list(CLASS_BASE_SCORES)
    for tree, class_index in zip(CLASS_TREES, CLASS_TREE_INFO):
        scores[class_index] += tree_predict(tree, features)
    return softmax(scores)


def predict_xgb_regressor(features):
    score = REG_BASE_SCORES[0]
    for tree in REG_TREES:
        score += tree_predict(tree, features)
    return max(0.0, min(500.0, score))


def season_from_month(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def make_input_row(form):
    month = int(form.get("month", ["1"])[0])
    day_of_month = int(form.get("day", ["15"])[0])
    day_of_year = max(1, min(365, (month - 1) * 30 + day_of_month))
    day_of_week = int(form.get("day_of_week", ["2"])[0])

    population = float(form.get("population", ["1000000"])[0])
    density = float(form.get("density", ["2500"])[0])

    return {
        "Number of Sites Reporting": float(form.get("sites", ["3"])[0]),
        "lat": float(form.get("lat", ["34.0522"])[0]),
        "lng": float(form.get("lng", ["-118.2437"])[0]),
        "population": population,
        "density": density,
        "log_population": math.log1p(population),
        "log_density": math.log1p(density),
        "month": month,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "is_weekend": 1 if day_of_week in (5, 6) else 0,
        "Defining Parameter": form.get("defining_parameter", ["PM2.5"])[0],
        "state_id": form.get("state_id", ["CA"])[0],
        "season": season_from_month(month),
    }


def predict_air_quality(form):
    row = make_input_row(form)
    class_features = transform_features(row, CLASSIFIER_CATEGORIES)
    reg_features = transform_features(row, REGRESSOR_CATEGORIES)

    probabilities = predict_xgb_classifier(class_features)
    predicted_aqi = predict_xgb_regressor(reg_features)
    # AQI categories are defined by numeric AQI bands, so the displayed result
    # stays consistent with the numeric prediction.
    category = aqi_category(predicted_aqi)

    ranked = sorted(zip(TARGET_CLASSES, probabilities), key=lambda item: item[1], reverse=True)
    return category, predicted_aqi, ranked


def select_options(values, selected):
    return "\\n".join(
        f'<option value="{html.escape(str(value))}" '
        f'{"selected" if str(value) == str(selected) else ""}>'
        f"{html.escape(str(value))}</option>"
        for value in values
    )


def render_page(result=None, error=None, values=None):
    values = values or {}
    pollutant_values = CLASSIFIER_CATEGORIES[CATEGORICAL_FEATURES.index("Defining Parameter")]
    state_values = CLASSIFIER_CATEGORIES[CATEGORICAL_FEATURES.index("state_id")]

    def field(name, default):
        return html.escape(str(values.get(name, default)))

    pollutant_options = select_options(pollutant_values, values.get("defining_parameter", "PM2.5"))
    state_options = select_options(state_values, values.get("state_id", "CA"))
    error_html = f'<div class="error" role="alert">{html.escape(error)}</div>' if error else ""

    result_html = """
      <section class="result result-empty">
        <span class="result-kicker">Forecast output</span>
        <h2>Ready when you are.</h2>
        <p>Choose a city and tune the scenario to see the expected AQI and category signal.</p>
        <div class="empty-meter"><span></span><span></span><span></span><span></span><span></span><span></span></div>
      </section>
    """
    if result:
        category, predicted_aqi, ranked = result
        tones = {
            "Good": ("#6ea977", "#e9f5ea", "0-50"),
            "Moderate": ("#c28a27", "#fff4dc", "51-100"),
            "Unhealthy for Sensitive Groups": ("#d16646", "#fff0e9", "101-150"),
            "Unhealthy": ("#b44c4d", "#fbe9e9", "151-200"),
            "Very Unhealthy": ("#76529d", "#f1eafa", "201-300"),
            "Hazardous": ("#7e4041", "#f5e7e7", "301-500"),
        }
        color, tint, band = tones[category]
        bars = "\\n".join(
            f'<div class="prob-row"><span>{html.escape(label)}</span><div class="bar"><i style="width:{prob * 100:.1f}%;background:{color}"></i></div><strong>{prob * 100:.0f}%</strong></div>'
            for label, prob in ranked
        )
        result_html = f"""
        <section class="result" style="--tone:{color};--tint:{tint};--score:{predicted_aqi / 5:.1f}">
          <div class="result-top"><span class="result-kicker">Forecast output</span><span class="band">AQI {band}</span></div>
          <div class="aqi-summary">
            <div><p class="aqi-label">Predicted air quality</p><h2>{html.escape(category)}</h2><p class="result-copy">Based on the predicted AQI band for this scenario.</p></div>
            <div class="aqi-ring"><div><strong>{predicted_aqi:.0f}</strong><span>AQI</span></div></div>
          </div>
          <div class="aqi-scale"><span style="background:#6ea977"></span><span style="background:#c28a27"></span><span style="background:#d16646"></span><span style="background:#b44c4d"></span><span style="background:#76529d"></span><span style="background:#7e4041"></span></div>
          <div class="probability-head"><span>Classifier confidence</span><span>Category signal</span></div>
          <div class="probabilities">{bars}</div>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atmosphere | US AQI Forecast</title>
  <style>
    :root {{ --ink:#102229; --muted:#5f747a; --paper:#f6f8f6; --line:#d8e2df; --teal:#087b79; --deep:#063f4d; --sun:#e6a24b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .hero {{ min-height:360px; color:white; background:linear-gradient(90deg,rgba(3,29,39,.94) 0%,rgba(3,48,61,.75) 44%,rgba(3,48,61,.08) 100%),url('/aqi-city-hero.png') center/cover; }}
    .hero-inner,.workspace,.footer-inner {{ width:min(1180px,calc(100% - 40px)); margin:auto; }}
    .nav {{ height:72px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.18); }}
    .brand {{ display:flex; align-items:center; gap:10px; font-weight:750; letter-spacing:0; }}
    .brand-mark {{ width:28px; height:28px; border:2px solid #b8ddd6; border-radius:50%; position:relative; }}
    .brand-mark:after {{ content:""; position:absolute; inset:6px; border-radius:50%; background:#e7b15e; }}
    .nav-note {{ font-size:13px; color:#c6d9d9; }}
    .hero-copy {{ padding:54px 0 62px; max-width:650px; }}
    .eyebrow {{ margin:0 0 14px; color:#c0e0db; font-size:12px; font-weight:750; letter-spacing:0; text-transform:uppercase; }}
    h1 {{ margin:0; max-width:640px; font-size:clamp(40px,6vw,68px); line-height:1.03; letter-spacing:0; }}
    .hero-copy p {{ max-width:525px; margin:18px 0 0; color:#d6e3e3; font-size:17px; line-height:1.55; }}
    .hero-stats {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:24px; }}
    .hero-stats span {{ padding:7px 10px; border:1px solid rgba(255,255,255,.22); background:rgba(1,24,32,.28); color:#eff8f7; font-size:12px; }}
    .workspace {{ display:grid; grid-template-columns:minmax(0,1.06fr) minmax(360px,.94fr); gap:24px; margin-top:-30px; padding-bottom:34px; align-items:start; }}
    .tool,.result {{ border:1px solid var(--line); background:#fff; border-radius:8px; box-shadow:0 16px 38px rgba(21,47,52,.10); }}
    .tool {{ padding:28px; }}
    .panel-head {{ display:flex; align-items:start; justify-content:space-between; gap:16px; margin-bottom:26px; }}
    .panel-head h2,.result h2 {{ margin:0; font-size:25px; letter-spacing:0; }}
    .panel-head p {{ margin:6px 0 0; color:var(--muted); font-size:14px; line-height:1.45; }}
    .step {{ width:30px; height:30px; display:grid; place-items:center; border-radius:50%; background:#e5f1ef; color:var(--teal); font-size:13px; font-weight:800; }}
    .fields {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px 18px; }}
    .field {{ display:grid; gap:8px; }}
    .field.wide {{ grid-column:1/-1; }}
    label,.field-label {{ font-size:13px; font-weight:750; color:#284046; }}
    input,select {{ min-height:44px; width:100%; border:1px solid #cddbd8; border-radius:5px; padding:10px 11px; background:#fff; color:var(--ink); font:inherit; outline:none; }}
    input:focus,select:focus {{ border-color:var(--teal); box-shadow:0 0 0 3px rgba(8,123,121,.12); }}
    .range-line {{ display:flex; align-items:center; gap:12px; }}
    input[type=range] {{ min-height:24px; padding:0; accent-color:var(--teal); cursor:pointer; }}
    output {{ min-width:70px; color:var(--teal); font-weight:800; font-size:13px; text-align:right; }}
    .field-hint {{ margin:0; color:var(--muted); font-size:12px; line-height:1.35; }}
    .button-row {{ display:flex; align-items:center; justify-content:space-between; gap:16px; border-top:1px solid var(--line); margin-top:28px; padding-top:20px; }}
    .button-row small {{ color:var(--muted); line-height:1.4; }}
    button {{ min-height:46px; border:0; border-radius:5px; padding:0 18px; background:var(--teal); color:#fff; font:inherit; font-weight:800; cursor:pointer; box-shadow:0 8px 15px rgba(8,123,121,.2); }}
    button:hover {{ background:#056865; }}
    .error {{ border-left:3px solid #b94a48; margin-bottom:18px; padding:10px 12px; background:#fff1f0; color:#7a2f2d; font-size:13px; }}
    .result {{ min-height:485px; padding:27px; }}
    .result-top,.probability-head {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
    .result-kicker {{ color:var(--muted); font-size:11px; font-weight:800; letter-spacing:0; text-transform:uppercase; }}
    .band {{ border:1px solid var(--tone,#b9c7c7); color:var(--tone,#47646a); background:var(--tint,#eff5f4); padding:5px 8px; font-size:11px; font-weight:800; }}
    .aqi-summary {{ display:flex; justify-content:space-between; gap:12px; align-items:center; margin:25px 0 20px; }}
    .aqi-label {{ margin:0 0 6px; color:var(--muted); font-size:13px; }}
    .result h2 {{ max-width:270px; color:var(--tone,#087b79); font-size:clamp(28px,4vw,39px); line-height:1.04; }}
    .result-copy {{ margin:9px 0 0; color:var(--muted); max-width:265px; font-size:13px; line-height:1.45; }}
    .aqi-ring {{ width:112px; height:112px; flex:none; display:grid; place-items:center; border-radius:50%; background:conic-gradient(var(--tone,#087b79) calc(var(--score,0)*1%),#e7eeec 0); }}
    .aqi-ring>div {{ width:88px; height:88px; display:grid; place-content:center; text-align:center; border-radius:50%; background:#fff; }}
    .aqi-ring strong {{ font-size:31px; line-height:1; }} .aqi-ring span {{ margin-top:3px; color:var(--muted); font-size:11px; font-weight:800; }}
    .aqi-scale {{ display:grid; grid-template-columns:repeat(6,1fr); gap:3px; margin:0 0 25px; }}
    .aqi-scale span {{ height:6px; }}
    .probability-head {{ margin-bottom:12px; color:var(--muted); font-size:12px; font-weight:750; }}
    .probabilities {{ display:grid; gap:12px; }}
    .prob-row {{ display:grid; grid-template-columns:minmax(128px,1fr) minmax(75px,1.3fr) 35px; gap:9px; align-items:center; font-size:12px; }}
    .prob-row span {{ line-height:1.15; }} .prob-row strong {{ font-size:12px; text-align:right; }}
    .bar {{ height:8px; overflow:hidden; background:#e7eeec; border-radius:99px; }} .bar i {{ display:block; height:100%; border-radius:99px; }}
    .result-empty {{ display:grid; align-content:center; background:linear-gradient(145deg,#fff,#f5faf8); }}
    .result-empty h2 {{ margin-top:9px; color:var(--deep); }} .result-empty p {{ max-width:310px; color:var(--muted); line-height:1.5; }}
    .empty-meter {{ display:grid; grid-template-columns:repeat(6,1fr); gap:4px; margin-top:28px; }} .empty-meter span {{ height:12px; background:#dbe7e4; }}
    .insight {{ border-top:1px solid var(--line); background:#eef4f1; }}
    .insight-inner {{ width:min(1180px,calc(100% - 40px)); margin:auto; display:grid; grid-template-columns:repeat(3,1fr); gap:28px; padding:24px 0; }}
    .insight b {{ display:block; margin-bottom:4px; font-size:14px; }} .insight span {{ color:var(--muted); font-size:12px; line-height:1.45; }}
    footer {{ background:#0b232b; color:#b8c8c8; }} .footer-inner {{ display:flex; justify-content:space-between; gap:16px; padding:18px 0; font-size:12px; }}
    @media (max-width:820px) {{ .workspace {{ grid-template-columns:1fr; margin-top:-18px; }} .hero {{ min-height:330px; }} .hero-copy {{ padding:42px 0 48px; }} }}
    @media (max-width:560px) {{ .hero-inner,.workspace,.footer-inner,.insight-inner {{ width:min(100% - 28px,1180px); }} .nav-note {{ display:none; }} .hero-copy {{ padding-top:36px; }} .fields {{ grid-template-columns:1fr; gap:17px; }} .tool,.result {{ padding:21px; }} input,select {{ min-height:48px; font-size:16px; }} input[type=range] {{ min-height:34px; }} .range-line {{ gap:8px; }} .button-row {{ align-items:stretch; flex-direction:column; }} button {{ width:100%; min-height:50px; }} .aqi-summary {{ align-items:flex-start; }} .aqi-ring {{ width:95px; height:95px; }} .aqi-ring>div {{ width:75px; height:75px; }} .aqi-ring strong {{ font-size:26px; }} .prob-row {{ grid-template-columns:minmax(110px,1fr) minmax(55px,1fr) 31px; }} .insight-inner {{ grid-template-columns:1fr; gap:14px; }} .footer-inner {{ flex-direction:column; }} }}
  </style>
</head>
<body>
  <section class="hero">
    <div class="hero-inner">
      <nav class="nav"><div class="brand"><span class="brand-mark"></span>Atmosphere</div><span class="nav-note">US air-quality forecasting studio</span></nav>
      <div class="hero-copy"><p class="eyebrow">AQI scenario explorer</p><h1>See the air before you step outside.</h1><p>Explore a historical AQI forecast using location, season, pollutant, and monitoring context.</p><div class="hero-stats"><span>US historical data</span><span>1980-2022 observations</span><span>XGBoost models</span></div></div>
    </div>
  </section>
  <main class="workspace">
    <form class="tool" method="post">
      <div class="panel-head"><div><h2>Build a scenario</h2><p>Set the place and conditions you want to explore.</p></div><span class="step">01</span></div>
      {error_html}
      <div class="fields">
        <div class="field"><label for="state">State</label><select id="state" name="state_id">{state_options}</select></div>
        <div class="field"><label for="pollutant">Defining parameter</label><select id="pollutant" name="defining_parameter">{pollutant_options}</select></div>
        <div class="field"><label for="lat">Latitude</label><input id="lat" name="lat" type="number" min="-90" max="90" step="0.0001" value="{field('lat', 34.0522)}" required></div>
        <div class="field"><label for="lng">Longitude</label><input id="lng" name="lng" type="number" min="-180" max="180" step="0.0001" value="{field('lng', -118.2437)}" required></div>
        <div class="field"><label for="population">Population</label><input id="population" name="population" type="number" min="0" step="1" value="{field('population', 3898747)}" required></div>
        <div class="field"><label for="density">Density</label><input id="density" name="density" type="number" min="0" step="0.1" value="{field('density', 3200)}" required></div>
        <div class="field"><label for="weekday">Day of week</label><select id="weekday">{select_options(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], '')}</select><input type="hidden" id="weekday-value" name="day_of_week" value="{field('day_of_week', 2)}"></div>
        <div class="field"><label for="month">Month</label><div class="range-line"><input id="month" name="month" type="range" min="1" max="12" value="{field('month', 8)}"><output id="month-label"></output></div></div>
        <div class="field"><label for="day">Day of month</label><div class="range-line"><input id="day" name="day" type="range" min="1" max="31" value="{field('day', 15)}"><output id="day-label"></output></div></div>
        <div class="field wide"><label for="sites">Reporting sites</label><div class="range-line"><input id="sites" name="sites" type="range" min="1" max="25" value="{field('sites', 3)}"><output id="sites-label"></output></div><p class="field-hint">The number of monitoring locations contributing to the daily report.</p></div>
      </div>
      <div class="button-row"><small>Historical model estimate.<br>Not a real-time public-health alert.</small><button type="submit">Run forecast</button></div>
    </form>
    {result_html}
  </main>
  <section class="insight"><div class="insight-inner"><div><b>Chronological validation</b><span>Tested on observations later than the training period.</span></div><div><b>13.6 AQI MAE</b><span>Average absolute regression error on the holdout set.</span></div><div><b>Transparent by design</b><span>Category bands are derived directly from the predicted AQI value.</span></div></div></section>
  <footer><div class="footer-inner"><span>Atmosphere - US AQI portfolio project</span><span>For educational exploration only</span></div></footer>
  <script>
    const months=['January','February','March','April','May','June','July','August','September','October','November','December'];
    const weekdays=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
    const month=document.getElementById('month'), day=document.getElementById('day'), sites=document.getElementById('sites'), weekday=document.getElementById('weekday'), weekdayValue=document.getElementById('weekday-value');
    function sync() {{ document.getElementById('month-label').textContent=months[Number(month.value)-1]; document.getElementById('day-label').textContent=day.value; document.getElementById('sites-label').textContent=sites.value+' sites'; weekday.value=weekdays[Number(weekdayValue.value)]; }}
    [month,day,sites].forEach(item=>item.addEventListener('input',sync));
    weekday.addEventListener('change',()=>{{ weekdayValue.value=String(weekdays.indexOf(weekday.value)); }});
    sync();
  </script>
</body>
</html>"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(render_page())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)
        values = {key: val[0] for key, val in form.items()}

        try:
            result = predict_air_quality(form)
            page = render_page(result=result, values=values)
        except Exception as exc:
            page = render_page(error=str(exc), values=values)

        self.respond(page)

    def respond(self, content):
        encoded = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return
