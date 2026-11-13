import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_NAME = "RetainIQ"
APP_TAGLINE = "Customer Retention Intelligence Platform"

st.set_page_config(page_title=f"{APP_NAME} | {APP_TAGLINE}", page_icon="🛒", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))


def IMG(name):
    """Find an image in the images/ folder or directly next to app.py."""
    for folder in ["images", ""]:
        path = os.path.join(BASE, folder, name)
        if os.path.exists(path):
            return path
    return None


# ---------- Colour palette (dark theme) ----------
BG, SURFACE, SURFACE2 = "#0B1020", "#151B30", "#1D2544"
TEXT, MUTED = "#EEF2FF", "#B4BDD8"
VIOLET, BLUE, TEAL, RED, AMBER, GREEN, GREY = (
    "#7C3AED", "#2563EB", "#0D9488", "#E11D48", "#D97706", "#059669", "#8792B0")
# lighter tones used for charts (readable on dark)
L_VIOLET, L_CYAN, L_GREEN, L_RED, L_AMBER = "#A78BFA", "#22D3EE", "#34D399", "#FB7185", "#FBBF24"

# ---------- Matplotlib: dark, transparent charts with light text ----------
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "savefig.transparent": True,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": "#3A4468",
    "xtick.color": TEXT, "ytick.color": TEXT, "legend.facecolor": "none",
    "legend.edgecolor": "none", "axes.grid": False,
})

# ---------- Custom CSS ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {{ --text:{TEXT}; --muted:{MUTED}; --surface:{SURFACE}; --surface2:{SURFACE2}; }}

/* ---------- Base: dark background, light text everywhere ---------- */
.stApp {{ background: {BG}; }}
.stApp::before {{            /* slow drifting glow behind the page */
    content:""; position:fixed; inset:-20%; pointer-events:none; opacity:.55;
    background:
      radial-gradient(40% 35% at 20% 20%, rgba(124,58,237,.28), transparent 70%),
      radial-gradient(35% 30% at 80% 15%, rgba(34,211,238,.18), transparent 70%),
      radial-gradient(40% 35% at 70% 85%, rgba(37,99,235,.22), transparent 70%);
    animation: drift 26s ease-in-out infinite alternate;
}}
@keyframes drift {{ from {{ transform: translate(-2%, -1%) scale(1); }} to {{ transform: translate(3%, 2%) scale(1.08); }} }}

[data-testid="stHeader"] {{ background: transparent; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p, .stApp li,
.stApp label, .stApp [data-testid="stMarkdownContainer"] {{
    color: var(--text); font-family: 'Manrope', sans-serif;
}}
.stApp h1, .stApp h2, .stApp h3 {{ font-weight: 800; letter-spacing: -.01em; }}
.stApp small, .stApp [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}
.stApp hr {{ border-color: #2A3355; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #121836 0%, #1B1446 100%);
                             border-right: 1px solid #2A3355; }}
[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div[data-testid="stMetricValue"],
[data-testid="stSidebar"] .brand, [data-testid="stSidebar"] .tagline {{ font-family: 'Manrope', sans-serif; }}
/* keep Streamlit's icon font so icon names never show up as plain text */
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{
    font-family: "Material Symbols Rounded", "Material Icons" !important; }}

/* ---------- Sidebar open/close button: fullscreen-style symbol ---------- */
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] button {{
    background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5'/></svg>") center/20px no-repeat,
                rgba(124,58,237,.55) !important;
    border-radius: 10px !important; width: 38px; height: 38px; transition: transform .2s ease, background-color .2s ease;
}}
[data-testid="stSidebarCollapseButton"] button:hover, [data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {{ transform: scale(1.1); }}
[data-testid="stSidebarCollapseButton"] button > *, [data-testid="stExpandSidebarButton"] > *,
[data-testid="stSidebarCollapsedControl"] button > * {{ display: none !important; }}

/* ---------- Animated radar logo ---------- */
.radar {{ position: relative; width: 104px; height: 104px; margin: 8px auto 6px auto; border-radius: 50%;
          border: 1px solid rgba(34,211,238,.6); overflow: hidden;
          background: repeating-radial-gradient(circle, transparent 0 15px, rgba(34,211,238,.35) 15px 16px),
                      radial-gradient(circle, rgba(34,211,238,.18), rgba(11,16,32,.9) 70%);
          box-shadow: 0 0 26px rgba(34,211,238,.35), inset 0 0 18px rgba(124,58,237,.4); }}
.radar::before {{ content: ""; position: absolute; inset: 0; border-radius: 50%;
          background: conic-gradient(from 0deg, transparent 0deg 285deg, rgba(34,211,238,.75) 359deg, transparent 360deg);
          animation: spin 3.5s linear infinite; }}
.radar .core {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
                font-size: 22px; z-index: 2; }}
.radar .blip {{ position: absolute; width: 8px; height: 8px; border-radius: 50%; z-index: 2; }}
.radar .b1 {{ top: 22%; left: 62%; background: #34D399; box-shadow: 0 0 8px #34D399; }}
.radar .b2 {{ top: 66%; left: 24%; background: #34D399; box-shadow: 0 0 8px #34D399; }}
.radar .b3 {{ top: 30%; left: 26%; background: #FB7185; box-shadow: 0 0 10px #FB7185; animation: blink 1.4s ease-in-out infinite; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes blink {{ 0%,100% {{ opacity: .25; transform: scale(.8); }} 50% {{ opacity: 1; transform: scale(1.5); }} }}
.brand {{ font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; margin: 6px 0 0 0; text-align: center; }}
.brand span {{ background: linear-gradient(90deg, {L_CYAN}, {L_VIOLET}); -webkit-background-clip: text;
               background-clip: text; -webkit-text-fill-color: transparent; }}
.tagline {{ color: {MUTED} !important; font-size: .85rem; margin-bottom: 8px; }}

/* ---------- Hero title ---------- */
.hero {{ margin: 4px 0 18px 0; }}
.hero .name {{ font-size: 2.6rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1;
               background: linear-gradient(90deg, {L_CYAN}, {L_VIOLET}, {L_CYAN}); background-size: 200% auto;
               -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
               animation: shimmer 6s linear infinite; }}
.hero .sub {{ color: var(--muted); font-size: 1.05rem; margin-top: 6px; }}
@keyframes shimmer {{ to {{ background-position: 200% center; }} }}

/* ---------- Cards and KPIs ---------- */
.stApp .card {{ background: var(--surface); border: 1px solid #2A3355; border-radius: 16px; padding: 20px 22px;
                border-top: 4px solid {VIOLET}; height: 100%; transition: transform .25s ease, box-shadow .25s ease; }}
.stApp .card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 28px rgba(0,0,0,.45); }}
.stApp .card h4 {{ margin: 0 0 8px 0; color: #FFFFFF; font-size: 1.05rem; }}
.stApp .card p {{ margin: 0; color: var(--muted); font-size: .95rem; line-height: 1.5; }}

.stApp .kpi {{ border-radius: 16px; padding: 18px; color: #FFFFFF; text-align: center;
               box-shadow: 0 6px 20px rgba(0,0,0,.4); animation: rise .7s ease both; }}
.stApp .kpi .v {{ font-size: 1.9rem; font-weight: 800; color: #FFFFFF; }}
.stApp .kpi .l {{ font-size: .9rem; color: #FFFFFF; opacity: .95; }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}

/* ---------- Risk banner ---------- */
.stApp .risk {{ border-radius: 16px; padding: 18px 22px; color: #FFFFFF; font-size: 1.25rem; font-weight: 800;
                margin: 8px 0 12px 0; }}
.stApp .risk.high {{ animation: pulse 1.8s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(251,113,133,.55); }}
                    50% {{ box-shadow: 0 0 0 14px rgba(251,113,133,0); }} }}

/* ---------- Animated gauge ---------- */
.gauge-wrap {{ text-align: center; }}
.gauge-wrap svg {{ width: 100%; max-width: 380px; overflow: visible; }}
.needle {{ transform-origin: 100px 100px; animation: sweep 1.6s cubic-bezier(.2,.8,.2,1) both; }}
@keyframes sweep {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(var(--deg)); }} }}
@property --num {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
.pct {{ font-size: 2.6rem; font-weight: 800; color: #FFFFFF; counter-reset: num var(--num);
        animation: count 1.6s cubic-bezier(.2,.8,.2,1) both; margin-top: -6px; }}
.pct::after {{ content: counter(num) "%"; }}
@keyframes count {{ from {{ --num: 0; }} to {{ --num: var(--target); }} }}

/* ---------- Comparison bars ---------- */
.bar-row {{ margin: 10px 0 16px 0; }}
.bar-label {{ display:flex; justify-content:space-between; color: var(--text); font-weight: 700; margin-bottom: 6px; }}
.bar-track {{ background: var(--surface2); border-radius: 30px; height: 16px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 30px; animation: grow 1.3s cubic-bezier(.2,.8,.2,1) both; }}
@keyframes grow {{ from {{ width: 0; }} }}

/* ---------- Buttons ---------- */
.stButton > button {{ background: linear-gradient(90deg, {VIOLET}, {BLUE}); color: #FFFFFF; border: 0;
                      border-radius: 30px; padding: .6rem 2.2rem; font-weight: 800; font-size: 1.05rem;
                      transition: transform .2s ease, box-shadow .2s ease; }}
.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 22px rgba(124,58,237,.5); color: #FFFFFF; }}
.stButton > button p {{ color: #FFFFFF !important; }}

/* ---------- Inputs (so labels AND values stay visible) ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {{
    background-color: var(--surface2) !important; border-color: #3A4468 !important; }}
[data-baseweb="input"] input, [data-baseweb="select"] div, [data-baseweb="select"] span,
[data-baseweb="select"] svg {{ color: var(--text) !important; fill: var(--text) !important; }}
[data-testid="stNumberInput"] button {{ background: var(--surface2); color: var(--text); }}
[data-testid="stNumberInput"] button svg {{ fill: var(--text); }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {{
    background-color: var(--surface2) !important; color: var(--text) !important; }}
[data-baseweb="popover"] li:hover {{ background-color: #2C3766 !important; }}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: var(--text) !important; }}

/* ---------- Tabs, expander, info boxes, table ---------- */
button[data-baseweb="tab"] {{ color: var(--muted); font-weight: 700; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: #FFFFFF; }}
[data-baseweb="tab-border"] {{ background-color: #2A3355; }}
[data-testid="stExpander"] details {{ background: var(--surface); border: 1px solid #2A3355; border-radius: 12px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: var(--text) !important; }}
[data-testid="stAlert"] {{ background: rgba(34,211,238,.10); border: 1px solid rgba(34,211,238,.35); border-radius: 12px; }}
[data-testid="stAlert"] * {{ color: var(--text) !important; }}
.dtable {{ overflow-x: auto; border: 1px solid #2A3355; border-radius: 12px; }}
.dtable table {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
.dtable th {{ background: var(--surface2); color: #FFFFFF; padding: 8px 10px; text-align: left; white-space: nowrap; }}
.dtable td {{ background: var(--surface); color: var(--text); padding: 7px 10px; border-top: 1px solid #2A3355; white-space: nowrap; }}

@media (prefers-reduced-motion: reduce) {{
    .stApp::before, .hero .name, .needle, .pct, .bar-fill, .kpi, .risk.high {{ animation: none !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ---------- Load model and data once ----------
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model.pkl"))


@st.cache_data
def load_data():
    for path in ["dataset.csv", os.path.join("data", "dataset.csv")]:
        full = os.path.join(BASE, path)
        if os.path.exists(full):
            return pd.read_csv(full)
    raise FileNotFoundError("dataset.csv not found")


model = load_model()
df = load_data()


def show_img(name, sidebar=False):
    """Show a full-width image (works on old and new Streamlit versions)."""
    target = st.sidebar if sidebar else st
    if IMG(name) is None:      # image missing: skip it instead of crashing
        return
    try:
        target.image(IMG(name), width="stretch")
    except Exception:
        target.image(IMG(name), use_container_width=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><div class="name">{title}</div><div class="sub">{subtitle}</div></div>',
                unsafe_allow_html=True)


def kpi(col, value, label, c1, c2):
    col.markdown(f'<div class="kpi" style="background:linear-gradient(135deg,{c1},{c2})">'
                 f'<div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def card(col, icon, title, text, color=VIOLET):
    col.markdown(f'<div class="card" style="border-top-color:{color}"><h4>{icon} {title}</h4><p>{text}</p></div>',
                 unsafe_allow_html=True)


def gauge_html(prob):
    """Animated semi-circle gauge (SVG + CSS): the needle sweeps and the number counts up."""
    def pt(p, r=80):
        a = np.pi * (1 - p)
        return 100 + r * np.cos(a), 100 - r * np.sin(a)

    def arc(p0, p1, colr):
        x0, y0 = pt(p0)
        x1, y1 = pt(p1)
        return (f'<path d="M {x0:.2f} {y0:.2f} A 80 80 0 0 1 {x1:.2f} {y1:.2f}" fill="none" '
                f'stroke="{colr}" stroke-width="20"/>')

    deg = prob * 180
    pct = round(prob * 100)
    svg = (
        '<svg viewBox="0 0 200 118">'
        + arc(0.0, 0.4, "#34D399") + arc(0.4, 0.7, "#FBBF24") + arc(0.7, 1.0, "#FB7185")
        + f'<g class="needle" style="--deg:{deg:.1f}deg">'
          '<line x1="100" y1="100" x2="28" y2="100" stroke="#FFFFFF" stroke-width="3.5" stroke-linecap="round"/></g>'
        + '<circle cx="100" cy="100" r="7" fill="#FFFFFF"/>'
        + '<text x="14" y="116" fill="#EEF2FF" font-size="9" font-family="Manrope,sans-serif">Safe</text>'
        + '<text x="186" y="116" fill="#EEF2FF" font-size="9" text-anchor="end" font-family="Manrope,sans-serif">Risk</text>'
        + '</svg>'
    )
    return (f'<div class="gauge-wrap">{svg}'
            f'<div class="pct" style="--target:{pct}"></div>'
            f'<div style="color:{MUTED};font-size:.9rem">Churn probability</div></div>')


def bar_html(label, value, colr):
    return (f'<div class="bar-row"><div class="bar-label"><span>{label}</span><span>{value:.1f}%</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{min(value, 100):.1f}%;background:{colr}"></div>'
            f'</div></div>')


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---------- Sidebar ----------
st.sidebar.markdown('<div class="radar"><span class="blip b1"></span><span class="blip b2"></span>'
                    '<span class="blip b3"></span><div class="core">🛒</div></div>', unsafe_allow_html=True)
st.sidebar.markdown(f'<div class="brand">🛒 <span>{APP_NAME}</span></div>'
                    f'<div class="tagline">{APP_TAGLINE}</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["🏠 Business Problem", "📊 Data Insights", "🔮 Prediction"])
st.sidebar.markdown("---")
st.sidebar.metric("Customers analysed", f"{len(df):,}")
st.sidebar.metric("Overall churn rate", f"{df['Churn'].mean() * 100:.1f}%")


# =====================================================
# PAGE 1 - BUSINESS PROBLEM
# =====================================================
if page == "🏠 Business Problem":
    show_img("banner_home.png")
    hero(f"🛒 {APP_NAME}", f"{APP_TAGLINE}: find the customers who are about to leave, before they leave.")

    a, b, c = st.columns(3)
    card(a, "⚠️", "Business Problem",
         "An online retailer is losing customers. Winning a new customer costs far more than keeping an "
         "existing one, so management wants to spot customers who are likely to leave.", RED)
    card(b, "🎯", "Analytics Objective",
         "Build a classification model that predicts whether a customer will churn (1) or stay (0), "
         "using behaviour, satisfaction and order history.", BLUE)
    card(c, "💡", "Business Decision",
         "High-risk customers get retention offers (discount, cashback, a call). Low-risk customers get normal "
         "communication, so no money is wasted.", GREEN)

    st.markdown("### 🗂️ Dataset at a glance")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, f"{len(df):,}", "Customers (rows)", VIOLET, BLUE)
    kpi(k2, df.shape[1], "Columns", BLUE, TEAL)
    kpi(k3, f"{df['Churn'].mean() * 100:.1f}%", "Churn rate", RED, AMBER)
    kpi(k4, "Random Forest", "Model used", TEAL, GREEN)

    st.markdown("")
    st.write("**Source:** Kaggle (Ankit Verma), E-commerce churn dataset.  \n"
             "**Target variable:** Churn (1 = customer left, 0 = customer stayed).")

    st.markdown("### 🧭 How this app works")
    s1, s2, s3, s4 = st.columns(4)
    card(s1, "1️⃣", "Learn", "The model studied thousands of past customers.", VIOLET)
    card(s2, "2️⃣", "Enter", "You type in a customer's details.", BLUE)
    card(s3, "3️⃣", "Predict", "The model gives a churn probability.", AMBER)
    card(s4, "4️⃣", "Act", "The app recommends a retention action.", GREEN)

    st.markdown("")
    with st.expander("👀 Preview the data"):
        st.markdown(f'<div class="dtable">{df.head(10).to_html(index=False, border=0)}</div>',
                    unsafe_allow_html=True)


# =====================================================
# PAGE 2 - DATA INSIGHTS
# =====================================================
elif page == "📊 Data Insights":
    show_img("banner_insights.png")
    hero("📊 Data Insights", "What the history of our customers tells us about who leaves.")

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, f"{len(df):,}", "Customers", VIOLET, BLUE)
    kpi(k2, f"{df['Churn'].mean() * 100:.1f}%", "Churn rate", RED, AMBER)
    kpi(k3, f"{df['Tenure'].mean():.1f}", "Average tenure (months)", BLUE, TEAL)
    kpi(k4, f"{df['CashbackAmount'].mean():.0f}", "Average cashback", TEAL, GREEN)
    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs(["😠 Complaints", "⭐ Satisfaction", "⏳ Tenure", "💳 Payment mode"])

    with tab1:
        rate = df.groupby("Complain")["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(["No complaint", "Complained"], rate.values, color=[L_GREEN, L_RED], width=0.5)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontweight="bold")
        ax.set_ylabel("Churn rate (%)"); style_ax(ax)
        st.pyplot(fig)
        st.info(f"Customers who complained churn at **{rate[1]:.1f}%** versus **{rate[0]:.1f}%** for those who "
                f"did not. Complaints are an early-warning sign, so resolve them fast.")

    with tab2:
        rate2 = df.groupby("SatisfactionScore")["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(rate2.index.astype(str), rate2.values, color=plt.cm.cool(np.linspace(0, 1, len(rate2))))
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xlabel("Satisfaction score (1 = low, 5 = high)"); ax.set_ylabel("Churn rate (%)")
        style_ax(ax)
        st.pyplot(fig)
        st.info(f"The highest churn is at satisfaction score **{rate2.idxmax()}** ({rate2.max():.1f}%). "
                f"Customers with low scores need attention.")

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for val, colr, lab in [(0, L_GREEN, "Stayed"), (1, L_RED, "Churned")]:
            ax.hist(df.loc[df["Churn"] == val, "Tenure"].dropna(), bins=20, alpha=.75, color=colr, label=lab)
        ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Customers"); ax.legend()
        style_ax(ax)
        st.pyplot(fig)
        med = df.groupby("Churn")["Tenure"].median()
        st.info(f"Median tenure of churned customers is **{med[1]:.0f} months** versus **{med[0]:.0f}** "
                f"for loyal customers. New customers are the most at risk.")

    with tab4:
        rate4 = (df.groupby("PreferredPaymentMode")["Churn"].mean() * 100).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.barh(rate4.index, rate4.values, color=plt.cm.plasma(np.linspace(0.35, 0.95, len(rate4))))
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xlabel("Churn rate (%)"); style_ax(ax)
        st.pyplot(fig)
        st.info(f"Highest churn by payment mode: **{rate4.index[-1]}** ({rate4.iloc[-1]:.1f}%). "
                f"Lowest: **{rate4.index[0]}** ({rate4.iloc[0]:.1f}%).")


# =====================================================
# PAGE 3 - PREDICTION
# =====================================================
else:
    show_img("banner_predict.png")
    hero("🔮 Predict Customer Churn", "Fill in the customer's details and press Predict.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 Customer behaviour")
        tenure = st.number_input("📅 Tenure (months with the company)", min_value=0, max_value=70, value=10)
        satisfaction = st.slider("⭐ Satisfaction score (1 = low, 5 = high)", 1, 5, 3)
        complain = st.selectbox("😠 Complained in the last month?", ["No", "Yes"])
        days_since = st.number_input("🕒 Days since last order", min_value=0, max_value=60, value=5)
        cashback = st.number_input("💰 Average cashback amount", min_value=0, max_value=400, value=150)

    with col2:
        st.markdown("#### 📦 Delivery & profile")
        city_tier = st.selectbox("🏙️ City tier", [1, 2, 3])
        warehouse = st.number_input("🚚 Distance warehouse to home (km)", min_value=1, max_value=150, value=15)
        addresses = st.number_input("📍 Number of saved addresses", min_value=1, max_value=25, value=3)
        payment = st.selectbox("💳 Preferred payment mode", sorted(df["PreferredPaymentMode"].unique()))
        marital = st.selectbox("💍 Marital status", sorted(df["MaritalStatus"].unique()))

    if st.button("🚀 Predict"):
        # Put the inputs in a table with the SAME column names used in training
        customer = pd.DataFrame([{
            "Tenure": tenure,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse,
            "SatisfactionScore": satisfaction,
            "NumberOfAddress": addresses,
            "Complain": 1 if complain == "Yes" else 0,
            "DaySinceLastOrder": days_since,
            "CashbackAmount": cashback,
            "PreferredPaymentMode": payment,
            "MaritalStatus": marital,
        }])

        prob = float(model.predict_proba(customer)[0][1])     # probability of churn

        st.markdown("---")
        st.subheader("📋 Result")
        left, right = st.columns([1, 1])

        with left:
            st.markdown(gauge_html(prob), unsafe_allow_html=True)

        with right:
            if prob >= 0.70:
                st.markdown(f'<div class="risk high" style="background:linear-gradient(90deg,{RED},#F43F5E)">'
                            f'🔴 HIGH RISK - likely to churn</div>', unsafe_allow_html=True)
                st.write("**Recommended action:** Call the customer, resolve any complaint and offer a strong "
                         "retention incentive (discount or extra cashback).")
            elif prob >= 0.40:
                st.markdown('<div class="risk" style="background:linear-gradient(90deg,#B45309,#D97706)">'
                            '🟠 MEDIUM RISK - may churn</div>', unsafe_allow_html=True)
                st.write("**Recommended action:** Send a personalised coupon or a re-engagement email and "
                         "monitor the next order.")
            else:
                st.markdown('<div class="risk" style="background:linear-gradient(90deg,#047857,#059669)">'
                            '🟢 LOW RISK - likely to stay</div>', unsafe_allow_html=True)
                st.write("**Recommended action:** No special action needed. Keep the normal loyalty and "
                         "rewards communication.")

        # Comparison bars: this customer vs the average customer
        avg_churn = df["Churn"].mean() * 100
        st.markdown("#### 📈 This customer vs the average customer")
        st.markdown(bar_html("Average customer", avg_churn, GREY)
                    + bar_html("This customer", prob * 100, L_RED if prob >= 0.5 else L_GREEN),
                    unsafe_allow_html=True)
