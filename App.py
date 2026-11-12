import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="E-Commerce Churn Predictor", page_icon="🛒", layout="wide")


# ---------- Load model and data once (not every click) ----------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv")


model = load_model()
df = load_data()

# ---------- Sidebar navigation ----------
st.sidebar.title("🛒 Churn Predictor")
page = st.sidebar.radio("Go to page", ["1. Business Problem", "2. Data Insights", "3. Prediction"])


# =====================================================
# PAGE 1 - BUSINESS PROBLEM
# =====================================================
if page == "1. Business Problem":
    st.title("🛒 E-Commerce Customer Churn Prediction")

    st.subheader("Business Problem")
    st.write(
        "An online retail company is losing customers. Winning a new customer costs far more "
        "than keeping an existing one, so management wants to find customers who are "
        "**likely to leave** before they actually leave."
    )

    st.subheader("Analytics Objective")
    st.write(
        "Build a classification model that predicts whether a customer will churn "
        "(**Churn = 1**) or stay (**Churn = 0**). Customers with a high churn risk can be "
        "targeted with retention offers such as discounts, cashback or a service call."
    )

    st.subheader("Dataset Information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers (rows)", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Overall churn rate", f"{df['Churn'].mean() * 100:.1f}%")
    st.write(
        "**Source:** Kaggle (Ankit Verma), E-commerce churn dataset.  \n"
        "**Target variable:** Churn (1 = customer left, 0 = customer stayed).  \n"
        "**Model used:** Random Forest Classifier."
    )
    with st.expander("Preview the data"):
        st.dataframe(df.head(10))


# =====================================================
# PAGE 2 - DATA INSIGHTS
# =====================================================
elif page == "2. Data Insights":
    st.title("📊 Data Insights")

    # Key statistics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['Churn'].mean() * 100:.1f}%")
    c3.metric("Average tenure (months)", f"{df['Tenure'].mean():.1f}")
    c4.metric("Average cashback", f"{df['CashbackAmount'].mean():.0f}")

    left, right = st.columns(2)

    # Chart 1 - churn by complaint
    with left:
        st.subheader("Churn rate by complaint")
        rate = df.groupby("Complain")["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(["No complaint", "Complained"], rate.values, color=["#4C9F70", "#E4572E"])
        ax.set_ylabel("Churn rate (%)")
        st.pyplot(fig)
        st.info(f"Customers who complained churn at **{rate[1]:.1f}%** versus "
                f"**{rate[0]:.1f}%** for those who did not. Complaints are an early-warning sign.")

    # Chart 2 - churn by satisfaction
    with right:
        st.subheader("Churn rate by satisfaction score")
        rate2 = df.groupby("SatisfactionScore")["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=rate2.index, y=rate2.values, ax=ax, color="#5B8DEF")
        ax.set_xlabel("Satisfaction score (1 = low, 5 = high)")
        ax.set_ylabel("Churn rate (%)")
        st.pyplot(fig)
        worst = rate2.idxmax()
        st.info(f"The highest churn is at satisfaction score **{worst}** "
                f"({rate2.max():.1f}%). Low-satisfaction customers need attention.")

    # Chart 3 - tenure
    st.subheader("Tenure of churned vs loyal customers")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.boxplot(x="Churn", y="Tenure", data=df, ax=ax, color="#5B8DEF")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stayed", "Churned"])
    st.pyplot(fig)
    med = df.groupby("Churn")["Tenure"].median()
    st.info(f"Median tenure of churned customers is **{med[1]:.0f} months** versus "
            f"**{med[0]:.0f} months** for loyal customers. New customers are the most at risk.")


# =====================================================
# PAGE 3 - PREDICTION
# =====================================================
else:
    st.title("🔮 Predict Customer Churn")
    st.write("Enter the customer details below and click **Predict**.")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months with the company)", min_value=0, max_value=70, value=10)
        satisfaction = st.slider("Satisfaction score (1 = low, 5 = high)", 1, 5, 3)
        complain = st.selectbox("Complained in the last month?", ["No", "Yes"])
        days_since = st.number_input("Days since last order", min_value=0, max_value=60, value=5)
        cashback = st.number_input("Average cashback amount", min_value=0, max_value=400, value=150)

    with col2:
        city_tier = st.selectbox("City tier", [1, 2, 3])
        warehouse = st.number_input("Distance warehouse to home (km)", min_value=1, max_value=150, value=15)
        addresses = st.number_input("Number of saved addresses", min_value=1, max_value=25, value=3)
        payment = st.selectbox("Preferred payment mode", sorted(df["PreferredPaymentMode"].unique()))
        marital = st.selectbox("Marital status", sorted(df["MaritalStatus"].unique()))

    if st.button("Predict", type="primary"):
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

        prob = model.predict_proba(customer)[0][1]     # probability of churn

        st.subheader("Result")
        st.metric("Churn probability", f"{prob * 100:.1f}%")
        st.progress(min(float(prob), 1.0))

        if prob >= 0.70:
            st.error("🔴 HIGH risk - this customer is likely to churn.")
            st.write("**Recommended action:** Call the customer, resolve any complaint and offer "
                     "a strong retention incentive (discount or extra cashback).")
        elif prob >= 0.40:
            st.warning("🟠 MEDIUM risk - this customer may churn.")
            st.write("**Recommended action:** Send a personalised coupon or a re-engagement "
                     "email and monitor the next order.")
        else:
            st.success("🟢 LOW risk - this customer is likely to stay.")
            st.write("**Recommended action:** No special action needed. Keep the normal loyalty "
                     "and rewards communication.")

        # Visualization: this customer vs the average customer
        avg_churn = df["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Average customer", "This customer"], [avg_churn, prob * 100],
               color=["#9AA5B1", "#E4572E" if prob >= 0.5 else "#4C9F70"])
        ax.set_ylabel("Churn probability (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)