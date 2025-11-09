import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "parkinsons_model_22.pkl"
TEST_CSV = "parkinsons_unseen_test.csv"

# ------------------------
# LOAD MODEL
# ------------------------
model = joblib.load(MODEL_PATH)
print("✅ Model loaded.")

# ------------------------
# LOAD HIDDEN TEST SET
# ------------------------
test_df = pd.read_csv(TEST_CSV)
X_test = test_df.drop(columns=["status"])
y_test = test_df["status"]
print(f"✅ Loaded hidden test set: {X_test.shape[0]} samples.")

# ------------------------
# PREDICT
# ------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of Parkinson's

# ------------------------
# EVALUATION METRICS
# ------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy on hidden test set: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))

# Risk levels
risk_levels = []
for p in y_prob:
    if p < 0.3:
        risk_levels.append("Low Risk")
    elif p < 0.7:
        risk_levels.append("Moderate Risk")
    else:
        risk_levels.append("High Risk")

test_df_eval = X_test.copy()
test_df_eval["True Label"] = y_test
test_df_eval["Predicted Label"] = y_pred
test_df_eval["Probability (%)"] = np.round(y_prob * 100, 2)
test_df_eval["Risk Level"] = risk_levels

# Show first 10 rows for verification
print("\nSample Predictions:")
print(test_df_eval.head(10))

# Optional: save evaluation results to CSV
test_df_eval.to_csv("hidden_test_results.csv", index=False)
print("\n✅ Evaluation results saved to 'hidden_test_results.csv'")
