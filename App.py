import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

st.set_page_config(page_title="Churn Radar | E-Commerce", page_icon="🛒", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = lambda name: os.path.join(BASE, "images", name)

# ---------- Colour palette ----------
PURPLE, BLUE, GREEN, RED, ORANGE, GREY = "#6A11CB", "#2575FC", "#11998E", "#FF5E62", "#FF9966", "#9AA5B1"

# ---------- Custom CSS (colours, cards, buttons) ----------
st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(180deg, #f6f3ff 0%, #ffffff 40%); }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {PURPLE} 0%, {BLUE} 100%); }}
[data-testid="stSidebar"] * {{ color: white !important; }}
h1, h2, h3 {{ color: #2b2d6e; }}
.card {{ background: white; border-radius: 16px; padding: 20px 22px; box-shadow: 0 4px 16px rgba(80,60,160,.12);
         border-top: 5px solid {PURPLE}; height: 100%; }}
.card h4 {{ margin: 0 0 6px 0; color: #2b2d6e; }}
.card p {{ margin: 0; color: #444; font-size: 0.95rem; }}
.kpi {{ border-radius: 16px; padding: 18px; color: white; text-align: center; box-shadow: 0 4px 14px rgba(0,0,0,.15); }}
.kpi .v {{ font-size: 2rem; font-weight: 800; }}
.kpi .l {{ font-size: .9rem; opacity: .95; }}
.risk {{ border-radius: 16px; padding: 18px 22px; color: white; font-size: 1.25rem; font-weight: 700; margin: 8px 0 12px 0; }}
.stButton > button {{ background: linear-gradient(90deg, {PURPLE}, {BLUE}); color: white; border: 0; border-radius: 30px;
                      padding: .6rem 2.2rem; font-weight: 700; font-size: 1.05rem; }}
.stButton > button:hover {{ filter: brightness(1.1); color: white; }}
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
    try:
        target.image(IMG(name), width="stretch")
    except Exception:
        target.image(IMG(name), use_container_width=True)


def kpi(col, value, label, c1, c2):
    col.markdown(f'<div class="kpi" style="background:linear-gradient(135deg,{c1},{c2})">'
                 f'<div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def card(col, icon, title, text, color=PURPLE):
    col.markdown(f'<div class="card" style="border-top-color:{color}"><h4>{icon} {title}</h4><p>{text}</p></div>',
                 unsafe_allow_html=True)


def gauge(prob):
    """Semi-circle speedometer showing churn probability."""
    fig, ax = plt.subplots(figsize=(5, 2.9))
    for start, end, colr in [(0, 40, "#38EF7D"), (40, 70, "#FFC837"), (70, 100, RED)]:
        ax.add_patch(Wedge((0, 0), 1, 180 - end * 1.8, 180 - start * 1.8, width=0.3, color=colr))
    angle = np.radians(180 - prob * 100 * 1.8)
    ax.plot([0, 0.85 * np.cos(angle)], [0, 0.85 * np.sin(angle)], color="#2b2d6e", lw=4)
    ax.add_patch(plt.Circle((0, 0), 0.06, color="#2b2d6e"))
    ax.text(0, -0.28, f"{prob * 100:.0f}%", ha="center", fontsize=24, fontweight="bold", color="#2b2d6e")
    ax.text(-1, -0.08, "Safe", ha="center", fontsize=9)
    ax.text(1, -0.08, "Risk", ha="center", fontsize=9)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.45, 1.1); ax.axis("off")
    fig.patch.set_alpha(0)
    return fig


# ---------- Sidebar ----------
show_img("logo.png", sidebar=True)
st.sidebar.title("Churn Radar")
st.sidebar.caption("Know who is leaving before they leave")
page = st.sidebar.radio("Navigate", ["🏠 Business Problem", "📊 Data Insights", "🔮 Prediction"])
st.sidebar.markdown("---")
st.sidebar.metric("Customers analysed", f"{len(df):,}")
st.sidebar.metric("Overall churn rate", f"{df['Churn'].mean() * 100:.1f}%")


# =====================================================
# PAGE 1 - BUSINESS PROBLEM
# =====================================================
if page == "🏠 Business Problem":
    show_img("banner_home.png")
    st.title("🛒 E-Commerce Customer Churn Prediction")
    st.write("Finding the customers who are about to leave, **before** they leave.")

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
    kpi(k1, f"{len(df):,}", "Customers (rows)", PURPLE, BLUE)
    kpi(k2, df.shape[1], "Columns", BLUE, GREEN)
    kpi(k3, f"{df['Churn'].mean() * 100:.1f}%", "Churn rate", RED, ORANGE)
    kpi(k4, "Random Forest", "Model used", GREEN, "#38EF7D")

    st.markdown("")
    st.write("**Source:** Kaggle (Ankit Verma), E-commerce churn dataset.  \n"
             "**Target variable:** Churn (1 = customer left, 0 = customer stayed).")

    st.markdown("### 🧭 How this app works")
    s1, s2, s3, s4 = st.columns(4)
    card(s1, "1️⃣", "Learn", "The model studied thousands of past customers.", PURPLE)
    card(s2, "2️⃣", "Enter", "You type in a customer's details.", BLUE)
    card(s3, "3️⃣", "Predict", "The model gives a churn probability.", ORANGE)
    card(s4, "4️⃣", "Act", "The app recommends a retention action.", GREEN)

    with st.expander("👀 Preview the data"):
        st.dataframe(df.head(10))


# =====================================================
# PAGE 2 - DATA INSIGHTS
# =====================================================
elif page == "📊 Data Insights":
    show_img("banner_insights.png")
    st.title("📊 Data Insights")

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, f"{len(df):,}", "Customers", PURPLE, BLUE)
    kpi(k2, f"{df['Churn'].mean() * 100:.1f}%", "Churn rate", RED, ORANGE)
    kpi(k3, f"{df['Tenure'].mean():.1f}", "Average tenure (months)", BLUE, GREEN)
    kpi(k4, f"{df['CashbackAmount'].mean():.0f}", "Average cashback", GREEN, "#38EF7D")
    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs(["😠 Complaints", "⭐ Satisfaction", "⏳ Tenure", "💳 Payment mode"])

    with tab1:
        rate = df.groupby("Complain")["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(["No complaint", "Complained"], rate.values, color=[GREEN, RED], width=0.5)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontweight="bold")
        ax.set_ylabel("Churn rate (%)"); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.info(f"Customers who complained churn at **{rate[1]:.1f}%** versus **{rate[0]:.1f}%** for those who "
                f"did not. Complaints are an early-warning sign, so resolve them fast.")

    with tab2:
        rate2 = df.groupby("SatisfactionScore")["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(rate2.index.astype(str), rate2.values, color=plt.cm.cool(np.linspace(0, 1, len(rate2))))
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xlabel("Satisfaction score (1 = low, 5 = high)"); ax.set_ylabel("Churn rate (%)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.info(f"The highest churn is at satisfaction score **{rate2.idxmax()}** ({rate2.max():.1f}%). "
                f"Customers with low scores need attention.")

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for val, colr, lab in [(0, GREEN, "Stayed"), (1, RED, "Churned")]:
            ax.hist(df.loc[df["Churn"] == val, "Tenure"].dropna(), bins=20, alpha=.65, color=colr, label=lab)
        ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Customers"); ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        med = df.groupby("Churn")["Tenure"].median()
        st.info(f"Median tenure of churned customers is **{med[1]:.0f} months** versus **{med[0]:.0f}** "
                f"for loyal customers. New customers are the most at risk.")

    with tab4:
        rate4 = (df.groupby("PreferredPaymentMode")["Churn"].mean() * 100).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.barh(rate4.index, rate4.values, color=plt.cm.plasma(np.linspace(0.2, 0.85, len(rate4))))
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xlabel("Churn rate (%)"); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.info(f"Highest churn by payment mode: **{rate4.index[-1]}** ({rate4.iloc[-1]:.1f}%). "
                f"Lowest: **{rate4.index[0]}** ({rate4.iloc[0]:.1f}%).")


# =====================================================
# PAGE 3 - PREDICTION
# =====================================================
else:
    show_img("banner_predict.png")
    st.title("🔮 Predict Customer Churn")
    st.write("Fill in the customer's details and press **Predict**.")

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
            st.pyplot(gauge(prob))

        with right:
            if prob >= 0.70:
                st.markdown(f'<div class="risk" style="background:linear-gradient(90deg,{RED},{ORANGE})">'
                            f'🔴 HIGH RISK - likely to churn</div>', unsafe_allow_html=True)
                st.write("**Recommended action:** Call the customer, resolve any complaint and offer a strong "
                         "retention incentive (discount or extra cashback).")
            elif prob >= 0.40:
                st.markdown('<div class="risk" style="background:linear-gradient(90deg,#F7971E,#FFD200)">'
                            '🟠 MEDIUM RISK - may churn</div>', unsafe_allow_html=True)
                st.write("**Recommended action:** Send a personalised coupon or a re-engagement email and "
                         "monitor the next order.")
            else:
                st.markdown('<div class="risk" style="background:linear-gradient(90deg,#11998E,#38EF7D)">'
                            '🟢 LOW RISK - likely to stay</div>', unsafe_allow_html=True)
                st.write("**Recommended action:** No special action needed. Keep the normal loyalty and "
                         "rewards communication.")
                st.balloons()

        # Comparison chart: this customer vs the average customer
        avg_churn = df["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 2.6))
        bars = ax.barh(["Average customer", "This customer"], [avg_churn, prob * 100],
                       color=[GREY, RED if prob >= 0.5 else GREEN])
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontweight="bold")
        ax.set_xlim(0, 105); ax.set_xlabel("Churn probability (%)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
