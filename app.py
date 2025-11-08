import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

# Define the local file path for the dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
MODEL_PATH = 'parkinsons_model.pkl'

def train_and_save_model():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_URL)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}. Please ensure you have internet access.")
        return

    # 2. Select Features (Using a subset of key acoustic features for training)
    # The full dataset has 22 features, we use a few to demonstrate the process.
    # For higher accuracy, use all features or run feature selection.
    features = [
        'MDVP:Fo(Hz)',  # Fundamental Frequency (Pitch)
        'MDVP:Jitter(%)',
        'MDVP:Shimmer',
        'MDVP:APQ',     # Amplitude Perturbation Quotient (Shimmer related)
        'NHR',          # Noise-to-Harmonics Ratio
        'HNR'           # Harmonics-to-Noise Ratio
    ]
    
    # Target label: 'status' (0 = healthy, 1 = Parkinson's)
    X = df[features]
    y = df['status']

    # 3. Preprocessing: Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the fitted scaler alongside the model, as it must be applied to live input
    joblib.dump(scaler, 'parkinsons_scaler.pkl')

    # 4. Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_scaled, y)

    # 5. Save Model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel trained and saved as {MODEL_PATH}")
    print(f"Scaler saved as parkinsons_scaler.pkl")
    print(f"Model trained on {len(features)} features. Ensure app.py extracts the same {len(features)} features.")

if __name__ == "__main__":
    train_and_save_model()