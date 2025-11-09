import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load hidden test set
test_df = pd.read_csv("parkinsons_unseen_test.csv")
X_test = test_df.drop(columns=["status"])
y_test = test_df["status"]

# Load trained model
model = joblib.load("parkinsons_model_22.pkl")

# Predict
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on hidden test set: {accuracy*100:.2f}%")
