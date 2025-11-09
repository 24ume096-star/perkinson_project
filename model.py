import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

MODEL_PATH = "parkinsons_model_22.pkl"

# Load Data
df = pd.read_csv(DATA_URL)

# Drop name column
df = df.drop(columns=["name"])

# The dataset already contains 22 voice biomarkers excluding name & status.
# So we select everything except the target label:
X = df.drop(columns=["status"])
y = df["status"]

# Build Pipeline: Scaler + RandomForest
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Model
pipeline.fit(X_train, y_train)

# Evaluate
pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"✅ Accuracy on test set: {acc:.4f}")

# Save trained model
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Model Saved as: {MODEL_PATH}")
