import base64
import html
import pickle
from datetime import datetime
from pathlib import Path
 
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
 
 
st.set_page_config(
    page_title="Telvex Churn Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
)
 
APP_DIR = Path(__file__).resolve().parent
 
MODEL_CANDIDATES = [
    APP_DIR / "churn_model.pkl",
    APP_DIR / "old churn model.pkl",
    APP_DIR / "old_churn_model.pkl",
]
 
 
CUSTOMERS = pd.DataFrame(
    [
        {"CustomerID": "TLV1001", "Name": "Riya Mehta", "City": "Mumbai", "tenure": 8, "MonthlyCharges": 999, "TotalCharges": 7992, "Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"},
        {"CustomerID": "TLV1002", "Name": "Kabir Shah", "City": "Delhi", "tenure": 46, "MonthlyCharges": 649, "TotalCharges": 29854, "Contract": "Two year", "InternetService": "DSL", "PaymentMethod": "Credit card"},
        {"CustomerID": "TLV1003", "Name": "Nisha Rao", "City": "Pune", "tenure": 38, "MonthlyCharges": 599, "TotalCharges": 22762, "Contract": "One year", "InternetService": "DSL", "PaymentMethod": "Bank transfer"},
        {"CustomerID": "TLV1004", "Name": "Dev Patel", "City": "Ahmedabad", "tenure": 5, "MonthlyCharges": 1150, "TotalCharges": 5750, "Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"},
        {"CustomerID": "TLV1005", "Name": "Aarav Khan", "City": "Lucknow", "tenure": 12, "MonthlyCharges": 869, "TotalCharges": 10428, "Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Mailed check"},
        {"CustomerID": "TLV1006", "Name": "Meera Iyer", "City": "Chennai", "tenure": 35, "MonthlyCharges": 769, "TotalCharges": 26915, "Contract": "One year", "InternetService": "DSL", "PaymentMethod": "Credit card"},
        {"CustomerID": "TLV1007", "Name": "Ishaan Verma", "City": "Jaipur", "tenure": 3, "MonthlyCharges": 1199, "TotalCharges": 3597, "Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"},
        {"CustomerID": "TLV1008", "Name": "Arjun Singh", "City": "Bhopal", "tenure": 28, "MonthlyCharges": 899, "TotalCharges": 25172, "Contract": "One year", "InternetService": "Fiber optic", "PaymentMethod": "Credit card"},
        {"CustomerID": "TLV1009", "Name": "Anika Das", "City": "Kolkata", "tenure": 19, "MonthlyCharges": 625, "TotalCharges": 11875, "Contract": "One year", "InternetService": "DSL", "PaymentMethod": "Bank transfer"},
        {"CustomerID": "TLV1010", "Name": "Vivaan Joshi", "City": "Surat", "tenure": 15, "MonthlyCharges": 837, "TotalCharges": 12555, "Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Mailed check"},
    ]
)
 
 
@st.cache_data
def get_background_css():
    candidates = [
        APP_DIR / "business_bg.jpg",
        APP_DIR / "business_bg1.jpg",
        APP_DIR / "assets" / "business_bg.jpg",
        APP_DIR / "assets" / "business_bg1.jpg",
    ]
    for path in candidates:
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f"url(data:image/jpeg;base64,{encoded})"
    return "radial-gradient(circle at 70% 20%, rgba(57,255,20,.12), transparent 35%)"
 
 
def find_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    pkl_files = list(APP_DIR.glob("*.pkl"))
    if len(pkl_files) == 1:
        return pkl_files[0]
    return None
 
 
@st.cache_resource
def load_model():
    model_path = find_model_path()
    if model_path is None:
        return None, "Model file nahi mili. Heuristic demo mode use ho raha hai."
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file), None
    except Exception as error:
        return None, f"Model load nahi hua: {error}. Heuristic demo mode use ho raha hai."
 
 
def get_model_features(model):
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if model is not None and getattr(model, "n_features_in_", 3) == 3:
        return ["tenure", "MonthlyCharges", "TotalCharges"]
    return ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]
 
 
def get_feature_value(feature, values):
    if feature in values:
        return values[feature]
    for base_feature, selected_value in values.items():
        prefix = f"{base_feature}_"
        if feature.startswith(prefix):
            encoded = feature.replace(prefix, "", 1)
            return int(str(encoded) == str(selected_value))
    return 0
 
 
def build_input_data(model, values):
    features = get_model_features(model)
    row = {feature: get_feature_value(feature, values) for feature in features}
    return pd.DataFrame([row], columns=features)
 
 
def heuristic_probability(values):
    score = 18
    if values["tenure"] <= 6:
        score += 35
    elif values["tenure"] <= 18:
        score += 22
    elif values["tenure"] <= 36:
        score += 8
    if values["MonthlyCharges"] >= 1000:
        score += 24
    elif values["MonthlyCharges"] >= 800:
        score += 14
    if values["Contract"] == "Month-to-month":
        score += 24
    elif values["Contract"] == "One year":
        score -= 8
    else:
        score -= 16
    if values["PaymentMethod"] == "Electronic check":
        score += 12
    if values["InternetService"] == "Fiber optic":
        score += 6
    return float(max(5, min(96, score)))
 
 
def predict_churn(values):
    # load_model is @st.cache_resource — returns instantly after first call
    model, model_warning = load_model()
    if model is None:
        probability = heuristic_probability(values)
        return int(probability >= 50), probability, model_warning
    try:
        data = build_input_data(model, values)
        if hasattr(model, "predict_proba"):
            classes = list(getattr(model, "classes_", [0, 1]))
            churn_index = classes.index(1) if 1 in classes else min(1, len(classes) - 1)
            probability = float(model.predict_proba(data)[0][churn_index]) * 100
        else:
            probability = heuristic_probability(values)
        prediction = model.predict(data)[0]
        return prediction, probability, model_warning
    except Exception as error:
        probability = heuristic_probability(values)
        return int(probability >= 50), probability, f"Model prediction issue: {error}. Demo fallback use hua."
 
 
def risk_label(probability):
    if probability < 40:
        return "Low Risk"
    if probability < 70:
        return "Medium Risk"
    return "High Risk"
 
 
def risk_color(probability):
    if probability < 40:
        return "#8cff6b"
    if probability < 70:
        return "#ffd21f"
    return "#ff4f78"
 
 
def recommendation(probability):
    if probability < 40:
        return ["Customer is stable.", "Offer loyalty reward points.", "Give cashback on next recharge.", "Invite to Telvex premium loyalty program."]
    if probability < 70:
        return ["Customer needs monitoring.", "Send personalized renewal offer.", "Recommend one-year contract upgrade.", "Trigger satisfaction check-in before billing cycle."]
    return ["High churn risk detected.", "Initiate retention call within 24 hours.", "Offer discount bundle or cashback.", "Escalate customer success follow-up."]
 
 
def revenue_loss(row, probability):
    return int(row["MonthlyCharges"] * 6 * probability / 100)
 
 
# ── Vectorized heuristic — runs once, zero Python loops ──────────
@st.cache_data(show_spinner=False)
def revenue_dataframe():
    df = CUSTOMERS.copy()
 
    model, _ = load_model()
 
    if model is not None:
        # Try batch model prediction first
        try:
            records = []
            for _, row in df.iterrows():
                values = customer_values(row)
                data = build_input_data(model, values)
                if hasattr(model, "predict_proba"):
                    classes = list(getattr(model, "classes_", [0, 1]))
                    churn_index = classes.index(1) if 1 in classes else min(1, len(classes) - 1)
                    prob = float(model.predict_proba(data)[0][churn_index]) * 100
                else:
                    prob = _vec_heuristic_single(row)
                records.append(prob)
            df["Probability"] = records
        except Exception:
            df["Probability"] = df.apply(_vec_heuristic_single, axis=1)
    else:
        # Pure vectorized — no Python loop, no function call overhead
        score = pd.Series(18.0, index=df.index)
        score += df["tenure"].map(lambda t: 35 if t <= 6 else (22 if t <= 18 else (8 if t <= 36 else 0)))
        score += df["MonthlyCharges"].map(lambda m: 24 if m >= 1000 else (14 if m >= 800 else 0))
        score += df["Contract"].map({"Month-to-month": 24, "One year": -8, "Two year": -16}).fillna(0)
        score += df["PaymentMethod"].map({"Electronic check": 12}).fillna(0)
        score += df["InternetService"].map({"Fiber optic": 6}).fillna(0)
        df["Probability"] = score.clip(5, 96)
 
    df["RevenueLoss"] = (df["MonthlyCharges"] * 6 * df["Probability"] / 100).astype(int)
    df["Risk"] = df["Probability"].apply(risk_label)
    return df
 
 
def _vec_heuristic_single(row):
    """Fallback single-row heuristic (only used when model errors on one row)."""
    score = 18
    t = row["tenure"]
    score += 35 if t <= 6 else (22 if t <= 18 else (8 if t <= 36 else 0))
    m = row["MonthlyCharges"]
    score += 24 if m >= 1000 else (14 if m >= 800 else 0)
    score += {"Month-to-month": 24, "One year": -8, "Two year": -16}.get(row["Contract"], 0)
    score += 12 if row["PaymentMethod"] == "Electronic check" else 0
    score += 6  if row["InternetService"] == "Fiber optic" else 0
    return float(max(5, min(96, score)))
 
 
def create_gauge(probability):
    color = risk_color(probability)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%", "font": {"size": 44, "color": "#8cff6b"}},
            title={"text": "Churn Probability", "font": {"color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#ffffff"},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(140,255,107,0.28)"},
                    {"range": [40, 70], "color": "rgba(255,210,31,0.30)"},
                    {"range": [70, 100], "color": "rgba(255,79,120,0.30)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"},
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig
 
 
def style_app():
    bg = get_background_css()
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"] {{
            background:
                linear-gradient(90deg, rgba(5,8,5,.94), rgba(5,9,12,.88)),
                {bg};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #ffffff;
            font-family: Inter, Arial, sans-serif;
        }}
 
        [data-testid="stHeader"] {{
            background: transparent;
        }}
 
        [data-testid="block-container"] {{
            max-width: 1350px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
 
        h1, h2, h3, p, label, span {{
            color: #ffffff;
        }}
 
        .brand {{
            font-size: 34px;
            font-weight: 900;
            margin-bottom: 80px;
        }}
 
        .brand span {{
            color: #8cff6b;
        }}
 
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 14px 20px;
            border: 1px solid rgba(140,255,107,.25);
            background: rgba(39, 78, 42, .50);
            color: #8cff6b;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
 
        .pill::before {{
            content: "";
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #8cff6b;
        }}
 
        /* ── HOME PAGE LAYOUT ── */
        .hero-wrapper {{
            display: flex;
            flex-direction: column;
            gap: 0;
        }}
 
        .hero-title {{
            margin-top: 28px;
            font-size: 40px;
            line-height: 1.30;
            font-weight: 900;
            color: #8cff6b;
        }}
 
        .hero-copy {{
            margin-top: 24px;
            max-width: 680px;
            font-size: 18px;
            line-height: 1.65;
            font-weight: 600;
            color: #d4ddd8;
        }}
 
        .hero-stats {{
            display: flex;
            gap: 28px;
            margin-top: 32px;
            flex-wrap: wrap;
        }}
 
        .stat-box {{
            padding: 16px 24px;
            border: 1px solid rgba(140,255,107,.20);
            background: rgba(20,40,25,.60);
            border-radius: 12px;
            min-width: 130px;
        }}
 
        .stat-num {{
            font-size: 28px;
            font-weight: 900;
            color: #8cff6b;
            line-height: 1;
        }}
 
        .stat-lbl {{
            font-size: 11px;
            color: #8ca096;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 6px;
        }}
 
        /* quote card on right */
        .quote-card {{
            padding: 30px;
            height: 100%;
            min-height: 420px;
            border: 1px solid rgba(140,255,107,.22);
            background: rgba(20, 58, 36, .75);
            border-radius: 18px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
 
        .quote-mark {{
            color: #8cff6b;
            font-size: 52px;
            line-height: 1;
            font-weight: 900;
        }}
 
        .quote-text {{
            font-size: 28px;
            line-height: 1.30;
            font-weight: 900;
            color: #f0f5f2;
            margin-top: 16px;
            flex: 1;
        }}
 
        .quote-footer {{
            margin-top: 24px;
        }}
 
        .muted-title {{
            color: #b8c1c7;
            font-weight: 900;
            font-size: 12px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
 
        .green-small {{
            color: #8cff6b;
            font-weight: 900;
            font-size: 13px;
            margin-top: 4px;
        }}
 
        /* ── BOTTOM TICKER BANNER ── */
        .ticker-banner {{
            width: 100%;
            margin-top: 52px;
            background: rgba(10, 28, 16, .85);
            border: 1px solid rgba(140,255,107,.20);
            border-radius: 14px;
            overflow: hidden;
            padding: 22px 0;
            position: relative;
        }}
 
        .ticker-label {{
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 110px;
            background: linear-gradient(90deg, rgba(10,28,16,1) 70%, transparent);
            display: flex;
            align-items: center;
            padding-left: 20px;
            font-size: 11px;
            font-weight: 900;
            letter-spacing: 2px;
            color: #8cff6b;
            text-transform: uppercase;
            z-index: 2;
        }}
 
        .ticker-track {{
            display: flex;
            width: max-content;
            animation: ticker 28s linear infinite;
            padding-left: 120px;
        }}
 
        .ticker-track:hover {{
            animation-play-state: paused;
        }}
 
        .ticker-item {{
            white-space: nowrap;
            font-size: 15px;
            font-weight: 700;
            color: #d0e8d4;
            padding: 0 40px;
            position: relative;
        }}
 
        .ticker-item::after {{
            content: "◆";
            color: #8cff6b;
            margin-left: 40px;
            font-size: 10px;
        }}
 
        @keyframes ticker {{
            from {{ transform: translateX(0); }}
            to   {{ transform: translateX(-50%); }}
        }}
 
        /* ── GLASS CARD & COMMON ── */
        .glass-card {{
            border: 1px solid rgba(160, 180, 190, .16);
            background: rgba(4, 10, 12, .72);
            border-radius: 18px;
            padding: 26px;
            box-shadow: 0 20px 60px rgba(0,0,0,.28);
        }}
 
        .big-number {{
            font-size: 34px;
            font-weight: 900;
            margin-top: 20px;
        }}
 
        .metric-label {{
            color: #9fc5e8;
            font-size: 12px;
            font-weight: 900;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }}
 
        .green {{ color: #8cff6b; }}
        .pink  {{ color: #ff4f78; }}
        .yellow {{ color: #ffd21f; }}
 
        .result-risk {{
            font-size: 46px;
            font-weight: 900;
            color: #8cff6b;
            margin-top: 70px;
        }}
 
        .stButton > button {{
            min-height: 56px;
            border-radius: 14px;
            border: 1px solid rgba(160, 180, 190, .18);
            background: rgba(5, 10, 12, .55);
            color: #ffffff;
            font-weight: 800;
        }}
 
        .stButton > button:hover {{
            border-color: rgba(140,255,107,.55);
            color: #8cff6b;
        }}
 
        div[data-testid="stForm"] button,
        .primary-button button,
        div[data-testid="stButton"] button[kind="primary"] {{
            background: linear-gradient(90deg, #8cff6b, #63dce6) !important;
            color: #03100d !important;
            border: none !important;
        }}
 
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stTextInput input {{
            background: rgba(24,25,36,.92);
            color: #ffffff;
            border: 0;
            border-radius: 8px;
        }}
 
        .recommend-box {{
            background: rgba(42, 117, 69, .55);
            color: #38ff77;
            padding: 18px;
            border-radius: 8px;
            font-weight: 700;
            margin: 20px 0;
        }}
 
        .divider {{
            height: 1px;
            background: rgba(255,255,255,.14);
            margin: 50px 0;
        }}
 
        .section-title {{
            font-size: 28px;
            font-weight: 900;
            margin: 28px 0 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
def go_to(page):
    st.session_state.page = page
 
 
def render_nav():
    left, b0, b1, b2, b3 = st.columns([4.2, 1.1, 1.1, 1.25, 1.65])
    with left:
        st.markdown('<div class="brand">Tel<span>vex</span></div>', unsafe_allow_html=True)
    with b0:
        if st.session_state.get("page", "Home") != "Home":
            if st.button("Back Home", use_container_width=True):
                go_to("Home")
    with b1:
        if st.button("Retention AI", use_container_width=True):
            go_to("Retention AI")
    with b2:
        if st.button("Revenue Risk", use_container_width=True):
            go_to("Revenue Risk")
    with b3:
        if st.button("Customer Intelligence", use_container_width=True):
            go_to("Customer Intelligence")
 
 
def render_home():
    render_nav()
 
    left, right = st.columns([1.35, 1])
 
    with left:
        st.markdown(
            """
            <div class="hero-wrapper">
                <div><span class="pill">Dark Fintech Telecom Intelligence</span></div>
                <div class="hero-title">
                    Predict churn.<br>
                    Protect revenue.<br>
                    Prioritize action.
                </div>
                <div class="hero-copy">
                    Telvex helps telecom teams detect churn signals, identify revenue
                    exposure, and trigger smarter retention decisions before customers leave.
                </div>
                <div class="hero-stats">
                    <div class="stat-box">
                        <div class="stat-num">10+</div>
                        <div class="stat-lbl">Customers Tracked</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num">AI</div>
                        <div class="stat-lbl">Powered Predictions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num">24h</div>
                        <div class="stat-lbl">Retention Alerts</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
    with right:
        st.markdown(
            """
            <div class="quote-card">
                <div>
                    <div class="quote-mark">''</div>
                    <div class="quote-text">
                        The best way<br>
                        to predict<br>
                        revenue is to<br>
                        protect the<br>
                        customers<br>
                        who create it.
                    </div>
                </div>
                <div class="quote-footer">
                    <div class="muted-title">Telvex Business Insight</div>
                    <div class="green-small">Retention-first growth strategy</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
    # ── Bottom ticker / animated business quotes banner ──
    ticker_items = [
        "Retaining a customer costs 5x less than acquiring a new one",
        "A 5% increase in retention can boost profits by up to 95%",
        "Churn is a revenue leak — plug it before it drains the pipeline",
        "Data-driven retention beats reactive customer service every time",
        "High monthly charges + short tenure = highest churn probability",
        "Month-to-month contracts carry 3x the churn risk of annual plans",
        "Electronic check users churn at significantly higher rates",
        "Act on the signal before the customer makes the decision to leave",
        "Revenue intelligence is not a dashboard — it is a decision engine",
        "Every high-risk customer is a retention conversation waiting to happen",
    ]
 
    # Duplicate for seamless loop
    all_items = ticker_items + ticker_items
    items_html = "".join(f'<span class="ticker-item">{item}</span>' for item in all_items)
 
    st.markdown(
        f"""
        <div class="ticker-banner">
            <div class="ticker-label">LIVE INTEL</div>
            <div class="ticker-track">
                {items_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
def customer_values(row):
    return {
        "tenure": int(row["tenure"]),
        "MonthlyCharges": float(row["MonthlyCharges"]),
        "TotalCharges": float(row["TotalCharges"]),
        "Contract": str(row["Contract"]),
        "InternetService": str(row["InternetService"]),
        "PaymentMethod": str(row["PaymentMethod"]),
    }
 
 
def render_hacker_terminal(probability, row):
    safe_name = html.escape(str(row["Name"]))
    risk = risk_label(probability).upper()
    offset = 565 - (565 * probability / 100)
 
    logs = [
        f"customer lock acquired: {row['CustomerID']}",
        f"risk score resolved at {probability:.1f} percent",
        "retention protocol standby",
        "billing signal inspected",
        "contract volatility indexed",
        "operator action channel open",
    ]
 
    log_html = ""
    now = datetime.now().strftime("%H:%M:%S")
    for line in logs + logs:
        log_html += f"<div class='log-line'>[{now}] {html.escape(line)}</div>"
 
    bars = [
        ("TENURE",    min(100, max(8, 100 - int(row["tenure"]) * 2))),
        ("BILLING",   min(100, int(row["MonthlyCharges"]) // 12)),
        ("CONTRACT",  88 if row["Contract"] == "Month-to-month" else 42),
        ("PAYMENT",   76 if row["PaymentMethod"] == "Electronic check" else 38),
        ("RETENTION", max(10, int(100 - probability))),
        ("LIVE RISK", int(probability)),
    ]
 
    bar_html = ""
    for index, (label, value) in enumerate(bars):
        bar_html += f"""
        <div class="signal-row">
            <span class="sig-lbl">{label}</span>
            <div class="bar-track"><div class="bar-fill" style="--w:{value}%;--d:{index * .12}s"></div></div>
            <span class="sig-val">{value}%</span>
        </div>
        """
 
    components.html(
        f"""
        <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
 
        body {{
            background: transparent;
            font-family: Consolas, "Courier New", monospace;
            color: #d9ffd9;
        }}
 
        .terminal {{
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(57,255,20,.52);
            background: rgba(0,0,0,.82);
            border-radius: 10px;
            padding: 28px 28px 20px;
            box-shadow: 0 0 34px rgba(57,255,20,.18), inset 0 0 24px rgba(57,255,20,.07);
        }}
 
        /* scan line */
        .terminal::before {{
            content: "";
            position: absolute;
            inset: -45%;
            background: linear-gradient(to bottom, transparent 42%, rgba(57,255,20,.18) 50%, transparent 58%);
            animation: scan 8s linear infinite;
            pointer-events: none;
        }}
 
        /* CRT lines */
        .terminal::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: repeating-linear-gradient(
                to bottom,
                rgba(57,255,20,.03), rgba(57,255,20,.03) 1px,
                transparent 3px, transparent 7px
            );
            pointer-events: none;
        }}
 
        @keyframes scan {{
            from {{ transform: translateY(-32%); }}
            to   {{ transform: translateY( 32%); }}
        }}
 
        /* ── TOP BAR ── */
        .top {{
            position: relative;
            z-index: 2;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            flex-wrap: wrap;
        }}
 
        .title {{
            color: #39ff14;
            font-size: 18px;
            font-weight: 900;
            text-shadow: 0 0 12px rgba(57,255,20,.55);
            line-height: 1.2;
        }}
 
        .sub {{
            margin-top: 6px;
            font-size: 12px;
            color: rgba(220,255,220,.75);
            line-height: 1.4;
        }}
 
        .clock {{
            color: #39ff14;
            text-align: right;
            font-size: 13px;
            white-space: nowrap;
            line-height: 1.5;
        }}
 
        .live {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            margin-top: 8px;
        }}
 
        .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #39ff14;
            box-shadow: 0 0 12px #39ff14;
            animation: pulse 1.4s ease-in-out infinite;
            flex-shrink: 0;
        }}
 
        @keyframes pulse {{
            0%,100% {{ transform: scale(.75); opacity: .55; }}
            50%      {{ transform: scale(1.25); opacity: 1;  }}
        }}
 
        /* ── MAIN GRID ── */
        .grid {{
            position: relative;
            z-index: 2;
            display: grid;
            grid-template-columns: 260px 1fr;
            gap: 28px;
            align-items: center;
            margin-top: 22px;
        }}
 
        /* ── RADAR RING ── */
        .radar {{
            width: 240px;
            height: 240px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
 
        .radar svg {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            transform: rotate(-90deg);
            filter: drop-shadow(0 0 10px rgba(57,255,20,.5));
        }}
 
        .bg-circle  {{ fill: transparent; stroke: rgba(57,255,20,.14); stroke-width: 10; }}
 
        .prog-circle {{
            fill: transparent;
            stroke: #39ff14;
            stroke-width: 10;
            stroke-linecap: round;
            stroke-dasharray: 565;
            stroke-dashoffset: {offset};
            animation: ring 1.2s ease-out forwards, radarPulse 3s ease-in-out infinite;
        }}
 
        @keyframes ring {{
            from {{ stroke-dashoffset: 565; }}
            to   {{ stroke-dashoffset: {offset}; }}
        }}
 
        @keyframes radarPulse {{
            0%,100% {{ opacity: .78; }}
            50%      {{ opacity: 1;   }}
        }}
 
        .radar-center {{
            position: relative;
            z-index: 2;
            text-align: center;
            line-height: 1;
        }}
 
        .radar-value {{
            font-size: 52px;
            font-weight: 900;
            color: #39ff14;
            text-shadow: 0 0 8px #39ff14, 0 0 18px rgba(57,255,20,.6);
            animation: flicker 3s infinite;
        }}
 
        @keyframes flicker {{
            0%,19%,21%,23%,80%,100% {{ opacity: 1;   }}
            20%,22%,81%             {{ opacity: .65; }}
        }}
 
        .radar-sub {{
            font-size: 11px;
            color: rgba(220,255,220,.65);
            margin-top: 6px;
            letter-spacing: 1px;
        }}
 
        /* ── SIGNAL BARS ── */
        .signal-section {{
            width: 100%;
        }}
 
        .signal-title {{
            color: rgba(220,255,220,.85);
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 14px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
 
        .signal-row {{
            display: grid;
            grid-template-columns: 90px minmax(0, 1fr) 42px;
            gap: 10px;
            align-items: center;
            margin: 10px 0;
            max-width: 100%;
        }}
 
        .sig-lbl {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .5px;
            color: rgba(220,255,220,.75);
            white-space: nowrap;
        }}
 
        .bar-track {{
            height: 10px;
            width: 100%;
            max-width: 100%;
            border: 1px solid rgba(57,255,20,.30);
            background: rgba(57,255,20,.06);
            overflow: hidden;
            border-radius: 2px;
        }}
 
        .bar-fill {{
            width: var(--w);
            height: 100%;
            background: linear-gradient(90deg, rgba(57,255,20,.35), #39ff14);
            box-shadow: 0 0 8px rgba(57,255,20,.45);
            transform-origin: left;
            animation: grow 1s ease-out both;
            animation-delay: var(--d);
        }}
 
        @keyframes grow {{
            from {{ transform: scaleX(0); }}
            to   {{ transform: scaleX(1); }}
        }}
 
        .sig-val {{
            font-size: 11px;
            color: #39ff14;
            text-align: right;
            white-space: nowrap;
        }}
 
        /* ── LOG STRIP ── */
        .logs {{
            position: relative;
            z-index: 2;
            height: 110px;
            overflow: hidden;
            margin-top: 20px;
            border-top: 1px solid rgba(57,255,20,.22);
            padding-top: 12px;
            font-size: 12px;
            color: rgba(220,255,220,.78);
        }}
 
        .log-lines {{
            animation: scroll 18s linear infinite;
        }}
 
        .log-line {{
            padding: 3px 0;
            line-height: 1.4;
        }}
 
        @keyframes scroll {{
            from {{ transform: translateY(0); }}
            to   {{ transform: translateY(-50%); }}
        }}
        </style>
 
        <div class="terminal">
            <!-- TOP -->
            <div class="top">
                <div>
                    <div class="title">CUSTOMER CHURN THREAT CONSOLE</div>
                    <div class="sub">TARGET: {safe_name.upper()} &nbsp;/&nbsp; STATUS: {risk} &nbsp;/&nbsp; MODEL LINK ACTIVE</div>
                    <div class="live"><span class="dot"></span> LIVE CUSTOMER RISK STREAM</div>
                </div>
                <div class="clock">SYSTEM CLOCK<br><span id="clock">--:--:--</span></div>
            </div>
 
            <!-- RADAR + BARS -->
            <div class="grid">
                <div class="radar">
                    <svg viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg">
                        <circle class="bg-circle"   cx="110" cy="110" r="90"></circle>
                        <circle class="prog-circle" cx="110" cy="110" r="90"></circle>
                    </svg>
                    <div class="radar-center">
                        <div class="radar-value">{probability:.0f}%</div>
                        <div class="radar-sub">CHURN RISK</div>
                    </div>
                </div>
 
                <div class="signal-section">
                    <div class="signal-title">Signal Intensity Map</div>
                    {bar_html}
                </div>
            </div>
 
            <!-- LOGS -->
            <div class="logs">
                <div class="log-lines">{log_html}</div>
            </div>
        </div>
 
        <script>
        function tickClock() {{
            document.getElementById("clock").textContent = new Date().toLocaleTimeString();
        }}
        tickClock();
        setInterval(tickClock, 1000);
        </script>
        """,
        height=540,
    )
 
 
def render_prediction_page():
    render_nav()
    st.markdown('<div class="section-title">Churn Prediction</div>', unsafe_allow_html=True)
 
    selected_id = st.selectbox(
        "Select customer",
        CUSTOMERS["CustomerID"].tolist(),
        index=7,
        format_func=lambda cid: f"{cid} - {CUSTOMERS.loc[CUSTOMERS.CustomerID == cid, 'Name'].iloc[0]}",
    )
 
    base_row = CUSTOMERS[CUSTOMERS["CustomerID"] == selected_id].iloc[0]
 
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name    = st.text_input("Customer Name", value=base_row["Name"])
            tenure  = st.number_input("Tenure", min_value=0, max_value=100, value=int(base_row["tenure"]))
        with c2:
            monthly  = st.number_input("Monthly Charges", min_value=0.0, value=float(base_row["MonthlyCharges"]))
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                    index=["Month-to-month", "One year", "Two year"].index(base_row["Contract"]))
        with c3:
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"],
                                    index=["Fiber optic", "DSL", "No"].index(base_row["InternetService"]))
            payment  = st.selectbox("Payment Method",
                                    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                                    index=["Electronic check", "Mailed check", "Bank transfer", "Credit card"].index(base_row["PaymentMethod"]))
        check = st.form_submit_button("Check Prediction", use_container_width=True)
 
    values = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
    }
 
    if check or "last_prediction_values" not in st.session_state:
        prediction, probability, warning = predict_churn(values)
        st.session_state.last_prediction_values = values
        st.session_state.last_probability       = probability
        st.session_state.last_prediction        = prediction
        st.session_state.last_customer          = {
            **base_row.to_dict(),
            "Name": name, "tenure": tenure, "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly, "Contract": contract,
            "InternetService": internet, "PaymentMethod": payment,
        }
        st.session_state.model_warning = warning
 
    row         = pd.Series(st.session_state.last_customer)
    probability = float(st.session_state.last_probability)
    recs        = recommendation(probability)
 
    if st.session_state.get("model_warning"):
        st.warning(st.session_state.model_warning)
 
    left, right = st.columns([1, 1.25])
 
    with left:
        st.markdown(
            f"""
            <div class="glass-card" style="min-height:330px;">
                <div class="pill">AI Prediction Result</div>
                <div class="result-risk" style="color:{risk_color(probability)};">{risk_label(probability)}</div>
                <p><b>Churn Probability:</b> {probability:.1f}%</p>
                <p><b>Customer:</b> {html.escape(str(row["Name"]))}</p>
                <p><b>Contract:</b> {row["Contract"]}</p>
                <p><b>Payment:</b> {row["PaymentMethod"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(create_gauge(probability), use_container_width=True)
 
    with right:
        st.markdown('<div class="section-title">Smart Retention Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="recommend-box">{recs[0]}</div>', unsafe_allow_html=True)
        for item in recs[1:]:
            st.markdown(f"- **{item}**")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Why this risk?</div>', unsafe_allow_html=True)
        st.markdown(f"- **Tenure:** {int(row['tenure'])} months")
        st.markdown(f"- **Contract:** {row['Contract']}")
        st.markdown(f"- **Monthly Charges:** Rs {float(row['MonthlyCharges']):,.0f}")
        st.markdown(f"- **Internet Service:** {row['InternetService']}")
        st.markdown(f"- **Payment Method:** {row['PaymentMethod']}")
 
    st.markdown('<div class="section-title">Live Threat Add-on</div>', unsafe_allow_html=True)
    render_hacker_terminal(probability, row)
 
 
def render_revenue_page():
    render_nav()
 
    top_left, top_right = st.columns([3.6, 1])
    with top_left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="pill">Revenue Intelligence</div>
                <br><br>
                <div style="font-size:44px;font-weight:900;">Revenue at Risk Dashboard</div>
                <br>
                <p style="font-size:16px;color:#b8c1c7;">
                    Estimate potential revenue loss from churn and identify customers who should be prioritized for retention.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.write("")
        st.write("")
        if st.button("Back to Home", use_container_width=True):
            go_to("Home")
 
    df          = revenue_dataframe().sort_values("RevenueLoss", ascending=False)
    selected_id = st.selectbox("Select CustomerID to view exact revenue risk", df["CustomerID"].tolist())
    selected    = df[df["CustomerID"] == selected_id].iloc[0]
 
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Selected Customer",  selected["CustomerID"],         selected["Name"],     "white"),
        ("Revenue Loss",       f"Rs {selected['RevenueLoss']:,.0f}", selected["Risk"], "white"),
        ("Monthly Charges",    f"Rs {selected['MonthlyCharges']:,.0f}", selected["City"], "white"),
        ("Churn Probability",  f"{selected['Probability']:.1f}%", "customer-level risk", "white"),
    ]
    for col, (label, value, sub, color) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">{label}</div>
                    <div class="big-number" style="color:{color};">{value}</div>
                    <br>
                    <div class="green-small">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
    m1, m2, m3  = st.columns(3)
    total_risk  = int(df["RevenueLoss"].sum())
    high_count  = int((df["Probability"] >= 70).sum())
    avg_risk    = float(df["Probability"].mean())
 
    with m1:
        st.markdown(f"""<div class="glass-card"><div class="metric-label">Total Revenue at Risk</div><div class="big-number pink">Rs {total_risk:,.0f}</div><br><div class="green-small">Six-month weighted exposure</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="glass-card"><div class="metric-label">High-Risk Customers</div><div class="big-number yellow">{high_count}</div><br><div class="green-small">Needs urgent retention</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="glass-card"><div class="metric-label">Average Risk</div><div class="big-number green">{avg_risk:.1f}%</div><br><div class="green-small">Across customer base</div></div>""", unsafe_allow_html=True)
 
    chart_col, pie_col = st.columns([1.25, 1])
 
    top10  = df.head(10).copy()
    colors = ["#63dce6" if cid == selected_id else risk_color(prob)
              for cid, prob in zip(top10["CustomerID"], top10["Probability"])]
 
    bar = go.Figure(
        go.Bar(
            x=top10["CustomerID"], y=top10["RevenueLoss"],
            marker_color=colors,
            text=[f"Rs {v:,.0f}" for v in top10["RevenueLoss"]],
            textposition="outside",
        )
    )
    bar.update_layout(
        title="Top 10 Risky Customers by Revenue Loss",
        yaxis_title="Revenue Loss (Rs)", xaxis_title="Customer ID",
        height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"}, margin=dict(l=20, r=20, t=60, b=40),
    )
    bar.update_xaxes(gridcolor="rgba(255,255,255,.08)")
    bar.update_yaxes(gridcolor="rgba(255,255,255,.18)")
 
    risk_counts = pd.cut(
        df["Probability"], bins=[0, 40, 70, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"], include_lowest=True,
    ).value_counts()
 
    donut = go.Figure(
        go.Pie(
            labels=risk_counts.index, values=risk_counts.values, hole=0.48,
            marker_colors=["#8cff6b", "#ffd21f", "#ff4f78"],
        )
    )
    donut.update_layout(
        title="Risk Distribution", height=390,
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ffffff"},
        margin=dict(l=20, r=20, t=60, b=40),
    )
 
    with chart_col:
        st.plotly_chart(bar, use_container_width=True)
    with pie_col:
        st.plotly_chart(donut, use_container_width=True)
 
 
def render_customer_intelligence():
    render_nav()
    st.markdown('<div class="section-title">Customer Intelligence</div>', unsafe_allow_html=True)
 
    df = revenue_dataframe().sort_values("Probability", ascending=False)
 
    st.markdown(
        """
        <div class="glass-card">
            <div class="pill">Customer Intelligence</div>
            <br><br>
            <div style="font-size:34px;font-weight:900;">Customer Threat Board</div>
            <p style="color:#b8c1c7;">A compact customer-level view for retention prioritization.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    st.dataframe(
        df[["CustomerID", "Name", "City", "tenure", "MonthlyCharges",
            "Contract", "PaymentMethod", "Probability", "RevenueLoss", "Risk"]],
        use_container_width=True,
        hide_index=True,
    )
 
 
# ── BOOTSTRAP ────────────────────────────────────────────────────
style_app()
 
# Pre-warm caches on first load so page switches are instant
load_model()          # loads & caches the model (or None) once
if "rev_df_ready" not in st.session_state:
    revenue_dataframe()                  # vectorized — <5 ms, cached after
    st.session_state.rev_df_ready = True
 
if "page" not in st.session_state:
    st.session_state.page = "Home"
 
page = st.session_state.page
 
if page == "Home":
    render_home()
elif page == "Retention AI":
    render_prediction_page()
elif page == "Revenue Risk":
    render_revenue_page()
else:
    render_customer_intelligence()
