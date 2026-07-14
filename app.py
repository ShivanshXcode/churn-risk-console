"""
Customer Churn Risk Console - Streamlit App
Run locally:  streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io -> connect repo
"""
import json
import numpy as np
import streamlit as st

st.set_page_config(page_title="Churn Risk Console", page_icon="📉", layout="wide")

with open("model_export.json") as f:
    MODEL = json.load(f)

FEATURE_LABELS = {
    "age": "Age", "partner": "Has partner", "dependents": "Has dependents",
    "tenure_months": "Tenure (months)", "tech_support": "Tech support add-on",
    "online_security": "Online security add-on", "streaming_service": "Streaming add-on",
    "paperless_billing": "Paperless billing", "monthly_charges": "Monthly charges",
    "avg_monthly_usage_gb": "Avg. monthly usage", "support_calls_6mo": "Support calls (6mo)",
    "late_payments_12mo": "Late payments (12mo)", "satisfaction_score": "Satisfaction score",
    "contract_One year": "One-year contract", "contract_Two year": "Two-year contract",
    "internet_service_Fiber optic": "Fiber internet", "internet_service_No": "No internet service",
    "payment_method_Credit card": "Pays by credit card",
    "payment_method_Electronic check": "Pays by e-check",
    "payment_method_Mailed check": "Pays by mailed check",
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tier_for(p):
    if p < 0.25:
        return "Low", "green"
    if p < 0.45:
        return "Medium", "orange"
    if p < 0.65:
        return "High", "orange"
    return "Critical", "red"


st.title("📉 Customer Churn Risk Console")
st.caption(f"Logistic regression trained on synthetic subscriber data — test AUC {MODEL['test_auc']}")

col_form, col_result = st.columns([1.3, 1])

with col_form:
    st.subheader("Customer Profile")
    c1, c2 = st.columns(2)
    age = c1.slider("Age", 18, 85, 42)
    tenure_months = c2.slider("Tenure (months)", 0, 72, 12)
    partner = c1.radio("Has partner", ["Yes", "No"], index=1, horizontal=True)
    dependents = c2.radio("Has dependents", ["Yes", "No"], index=1, horizontal=True)

    st.subheader("Contract & Billing")
    contract = st.radio("Contract type", ["Month-to-month", "One year", "Two year"], horizontal=True)
    payment_method = st.selectbox(
        "Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    paperless_billing = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    monthly_charges = st.slider("Monthly charges ($)", 18, 130, 70)

    st.subheader("Service")
    internet_service = st.radio("Internet service", ["Fiber optic", "DSL", "No"], horizontal=True)
    c3, c4, c5 = st.columns(3)
    tech_support = c3.radio("Tech support", ["Yes", "No"], index=1, horizontal=True)
    online_security = c4.radio("Online security", ["Yes", "No"], index=1, horizontal=True)
    streaming_service = c5.radio("Streaming add-on", ["Yes", "No"], index=1, horizontal=True)
    avg_monthly_usage_gb = st.slider("Avg. monthly usage (GB)", 0, 500, 90)

    st.subheader("Engagement & Support")
    c6, c7, c8 = st.columns(3)
    support_calls_6mo = c6.slider("Support calls (6mo)", 0, 10, 1)
    late_payments_12mo = c7.slider("Late payments (12mo)", 0, 10, 0)
    satisfaction_score = c8.slider("Satisfaction (1-5)", 1, 5, 3)

# --- build feature vector in exact model order ---
yn = lambda v: 1 if v == "Yes" else 0
raw = {
    "age": age, "partner": yn(partner), "dependents": yn(dependents),
    "tenure_months": tenure_months, "tech_support": yn(tech_support),
    "online_security": yn(online_security), "streaming_service": yn(streaming_service),
    "paperless_billing": yn(paperless_billing), "monthly_charges": monthly_charges,
    "avg_monthly_usage_gb": avg_monthly_usage_gb, "support_calls_6mo": support_calls_6mo,
    "late_payments_12mo": late_payments_12mo, "satisfaction_score": satisfaction_score,
    "contract_One year": 1 if contract == "One year" else 0,
    "contract_Two year": 1 if contract == "Two year" else 0,
    "internet_service_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "internet_service_No": 1 if internet_service == "No" else 0,
    "payment_method_Credit card": 1 if payment_method == "Credit card" else 0,
    "payment_method_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "payment_method_Mailed check": 1 if payment_method == "Mailed check" else 0,
}

z = MODEL["intercept"]
contributions = []
for i, name in enumerate(MODEL["feature_order"]):
    scaled = (raw[name] - MODEL["scaler_mean"][i]) / MODEL["scaler_scale"][i]
    contrib = MODEL["coefficients"][i] * scaled
    z += contrib
    contributions.append((name, contrib))

probability = sigmoid(z)
tier_label, tier_color = tier_for(probability)
contributions.sort(key=lambda x: abs(x[1]), reverse=True)

with col_result:
    st.subheader("Prediction")
    st.metric("Churn probability", f"{probability*100:.1f}%")
    st.markdown(f"**Risk tier:** :{tier_color}[{tier_label}]")
    st.progress(min(probability, 1.0))

    st.subheader("Top Signal Drivers")
    for name, contrib in contributions[:6]:
        direction = "⬆️ increases risk" if contrib > 0 else "⬇️ decreases risk"
        st.write(f"**{FEATURE_LABELS.get(name, name)}** — {direction}")
