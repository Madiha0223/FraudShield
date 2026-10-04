import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

data = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nClass distribution:")
print(data["Class"].value_counts())


# --------------------------------------------------
# 2. CREATE USER-FRIENDLY FEATURES
# --------------------------------------------------

print("\nCreating FraudShield features...")


# Convert Time into transaction hour
data["Hour"] = ((data["Time"] // 3600) % 24).astype(int)


# Log transformation of Amount
data["Amount_Log"] = np.log1p(data["Amount"])


# Median transaction amount
median_amount = data["Amount"].median()


# Amount compared with median
data["Amount_Ratio"] = (
    data["Amount"] / (median_amount + 0.01)
)


# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

features = [
    "Time",
    "Hour",
    "Amount",
    "Amount_Log",
    "Amount_Ratio"
]

X = data[features]
y = data["Class"]


print("\nFraudShield features:")
for feature in features:
    print("-", feature)


# --------------------------------------------------
# 4. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining FraudShield ML model...")


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=1
)

model.fit(X_train, y_train)


print("Model training completed!")


# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------

y_pred = model.predict(X_test)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 7. FIND A REAL HIGH-RISK FRAUD SAMPLE
# --------------------------------------------------

print("\nSearching for a real fraud sample...")


fraud_rows = data[data["Class"] == 1].copy()


# Calculate probabilities for real fraud samples
fraud_probabilities = model.predict_proba(
    fraud_rows[features]
)[:, 1]


fraud_rows["Fraud_Probability"] = (
    fraud_probabilities * 100
)


high_risk_frauds = fraud_rows[
    fraud_rows["Fraud_Probability"] >= 70
]


if len(high_risk_frauds) > 0:

    highest_risk = high_risk_frauds.loc[
        high_risk_frauds["Fraud_Probability"].idxmax()
    ]

    print("\n========================================")
    print("HIGH-RISK FRAUD SAMPLE FOUND")
    print("========================================")

    print(
        "Fraud Probability:",
        round(
            highest_risk["Fraud_Probability"],
            2
        ),
        "%"
    )

    print(
        "Actual Class:",
        highest_risk["Class"]
    )

    print(
        "Time:",
        highest_risk["Time"]
    )

    print(
        "Hour:",
        highest_risk["Hour"]
    )

    print(
        "Amount:",
        highest_risk["Amount"]
    )

    print("========================================")


else:

    print(
        "\nNo fraud sample above 70% "
        "was found."
    )


# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

model_package = {
    "model": model,
    "features": features,
    "median_amount": median_amount
}


joblib.dump(
    model_package,
    "model/fraud_model.pkl"
)


print("\n========================================")
print("FraudShield model saved successfully!")
print("Location: model/fraud_model.pkl")
print("========================================")