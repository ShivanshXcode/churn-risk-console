"""
Customer Churn Analysis & Predictive Model
Loads customer_churn.csv, explores key churn drivers, trains a logistic
regression + random forest model, evaluates them, and exports the logistic
regression's coefficients/scaling so they can power a lightweight in-browser
prediction app.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

df = pd.read_csv("data/customer_churn.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
print(f"Overall churn rate: {df['churn'].mean():.1%}\n")

# ---------------------------------------------------------------
# EDA: churn rate by key segments
# ---------------------------------------------------------------
print("Churn rate by contract type:")
print(df.groupby("contract")["churn"].mean().round(3).to_string(), "\n")

print("Churn rate by tech support:")
print(df.groupby("tech_support")["churn"].mean().round(3).to_string(), "\n")

print("Churn rate by internet service:")
print(df.groupby("internet_service")["churn"].mean().round(3).to_string(), "\n")

# tenure buckets
df["tenure_bucket"] = pd.cut(
    df["tenure_months"], bins=[-1, 6, 12, 24, 48, 100],
    labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49+mo"]
)
print("Churn rate by tenure bucket:")
print(df.groupby("tenure_bucket", observed=True)["churn"].mean().round(3).to_string(), "\n")

# ---------------------------------------------------------------
# Chart 1: churn rate by contract & tenure bucket
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

contract_rates = df.groupby("contract")["churn"].mean().reindex(
    ["Month-to-month", "One year", "Two year"])
axes[0].bar(contract_rates.index, contract_rates.values, color=["#C4574A", "#D9A441", "#4A7C6F"])
axes[0].set_title("Churn Rate by Contract Type")
axes[0].set_ylabel("Churn Rate")
for i, v in enumerate(contract_rates.values):
    axes[0].text(i, v + 0.01, f"{v:.0%}", ha="center", fontweight="bold")

tenure_rates = df.groupby("tenure_bucket", observed=True)["churn"].mean()
axes[1].bar(tenure_rates.index.astype(str), tenure_rates.values, color="#4A6FA5")
axes[1].set_title("Churn Rate by Tenure")
axes[1].set_ylabel("Churn Rate")
for i, v in enumerate(tenure_rates.values):
    axes[1].text(i, v + 0.01, f"{v:.0%}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("docs/chart_segments.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# Model prep: encode categoricals
# ---------------------------------------------------------------
model_df = df.copy()
binary_map = {"Yes": 1, "No": 0}
for col in ["partner", "dependents", "tech_support", "online_security",
            "streaming_service", "paperless_billing"]:
    model_df[col] = model_df[col].map(binary_map)

model_df = pd.get_dummies(model_df, columns=["contract", "internet_service", "payment_method"],
                           drop_first=True)

feature_cols = [c for c in model_df.columns if c not in
                ["customer_id", "churn", "tenure_bucket", "total_charges"]]

X = model_df[feature_cols].astype(float)
y = model_df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# Model 1: Logistic Regression (interpretable, powers the app)
# ---------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, C=1.0)
logreg.fit(X_train_s, y_train)
pred_proba_lr = logreg.predict_proba(X_test_s)[:, 1]
auc_lr = roc_auc_score(y_test, pred_proba_lr)

print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)
print(f"Test AUC: {auc_lr:.3f}")
print(classification_report(y_test, logreg.predict(X_test_s), digits=3))

# ---------------------------------------------------------------
# Model 2: Random Forest (benchmark / feature importance)
# ---------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, min_samples_leaf=20)
rf.fit(X_train, y_train)
pred_proba_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, pred_proba_rf)

print("=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)
print(f"Test AUC: {auc_rf:.3f}")
print(classification_report(y_test, rf.predict(X_test), digits=3))

# Feature importance chart (RF)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
plt.barh(importances.index[::-1], importances.values[::-1], color="#4A6FA5")
plt.title("Top 10 Churn Drivers (Random Forest Feature Importance)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("docs/chart_importance.png", dpi=140)
plt.close()

print("\nTop feature importances:")
print(importances.round(3).to_string())

# ---------------------------------------------------------------
# Export logistic regression coefficients + scaler stats to JSON
# so a lightweight browser app can reproduce predictions exactly.
# ---------------------------------------------------------------
export = {
    "feature_order": feature_cols,
    "coefficients": logreg.coef_[0].tolist(),
    "intercept": float(logreg.intercept_[0]),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "test_auc": round(auc_lr, 4),
    "train_rows": len(X_train),
}
with open("model_export.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nModel exported to model_export.json")
print("Feature columns:", feature_cols)
