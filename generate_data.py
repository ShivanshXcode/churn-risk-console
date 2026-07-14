"""
Generate a realistic synthetic customer churn dataset.
Modeled loosely on telecom/subscription-business churn patterns.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# --- Core demographic / account features ---
tenure_months = np.random.gamma(shape=2.0, scale=12, size=N).clip(0, 72).round().astype(int)
contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.27, 0.18]
)
age = np.random.normal(42, 14, N).clip(18, 85).round().astype(int)
partner = np.random.choice(["Yes", "No"], size=N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], size=N, p=[0.3, 0.7])

# --- Service features ---
internet_service = np.random.choice(
    ["Fiber optic", "DSL", "No"], size=N, p=[0.44, 0.34, 0.22]
)
monthly_charges = np.where(
    internet_service == "Fiber optic", np.random.normal(85, 15, N),
    np.where(internet_service == "DSL", np.random.normal(55, 12, N), np.random.normal(25, 8, N))
).clip(18, 130).round(2)

tech_support = np.random.choice(["Yes", "No"], size=N, p=[0.4, 0.6])
online_security = np.random.choice(["Yes", "No"], size=N, p=[0.38, 0.62])
streaming = np.random.choice(["Yes", "No"], size=N, p=[0.45, 0.55])
paperless_billing = np.random.choice(["Yes", "No"], size=N, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=N, p=[0.34, 0.19, 0.23, 0.24]
)

# --- Behavioral / engagement features ---
support_calls = np.random.poisson(1.2, N)
support_calls = np.where(tenure_months < 6, support_calls + np.random.poisson(0.8, N), support_calls)
late_payments = np.random.poisson(0.6, N)
avg_monthly_usage_gb = np.random.gamma(2, 40, N).clip(0, 500).round(1)
satisfaction_score = np.random.randint(1, 6, N)  # 1-5 survey score

total_charges = (monthly_charges * tenure_months + np.random.normal(0, 50, N)).clip(0, None).round(2)

# --- Build churn probability from a realistic logistic combination ---
# Coefficients chosen to reflect known real-world churn drivers
z = (
    -2.3
    + 1.4 * (contract == "Month-to-month")
    - 0.9 * (contract == "Two year")
    - 0.03 * tenure_months
    + 0.012 * monthly_charges
    + 0.35 * support_calls
    + 0.30 * late_payments
    - 0.45 * (tech_support == "Yes")
    - 0.35 * (online_security == "Yes")
    + 0.5 * (internet_service == "Fiber optic")
    + 0.25 * (payment_method == "Electronic check")
    - 0.5 * (satisfaction_score - 3)
    - 0.15 * (partner == "Yes")
    + np.random.normal(0, 0.6, N)  # noise
)
prob_churn = 1 / (1 + np.exp(-z))
churn = (np.random.random(N) < prob_churn).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST-{10000+i}" for i in range(N)],
    "age": age,
    "partner": partner,
    "dependents": dependents,
    "tenure_months": tenure_months,
    "contract": contract,
    "internet_service": internet_service,
    "tech_support": tech_support,
    "online_security": online_security,
    "streaming_service": streaming,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "avg_monthly_usage_gb": avg_monthly_usage_gb,
    "support_calls_6mo": support_calls,
    "late_payments_12mo": late_payments,
    "satisfaction_score": satisfaction_score,
    "churn": churn,
})

df.to_csv("data/customer_churn.csv", index=False)
print(f"Generated {len(df)} rows. Churn rate: {df['churn'].mean():.1%}")
print(df.head())
