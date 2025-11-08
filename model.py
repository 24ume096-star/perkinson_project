import streamlit as st
import librosa
import numpy as np
import joblib
from audio_recorder_streamlit import audio_recorder
import io
import soundfile as sf

MODEL_PATH = "parkinsons_model.pkl"
SCALER_PATH = "parkinsons_scaler.pkl"

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except:
        st.error("❌ Model or Scaler missing. Train model first.")
        return None, None

def extract_mfcc(audio_bytes, sr=16000):
    """Convert raw byte audio to MFCC feature vector."""
    try:
        # Convert bytes to wav buffer
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Decode audio
        y, file_sr = sf.read(audio_buffer)
        
        # If stereo → convert to mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Resample if necessary
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

        # Normalize audio
        y = librosa.util.normalize(y)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6)
        mfcc_mean = np.mean(mfccs, axis=1)
        return mfcc_mean.reshape(1, -1)

    except Exception as e:
        raise ValueError(f"Feature extraction failed: {e}")

def main():
    st.title("🎙️ Parkinson’s Voice Detection (MFCC-Based)")
    st.write("Hold **'Aaaaaah'** for 3 seconds and record.")

    model, scaler = load_resources()
    if model is None:
        return

    audio_bytes = audio_recorder(text="🎤 Click to Record", sample_rate=16000)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("🔍 Analyze Voice"):
            try:
                st.write("Extracting MFCC features...")
                features = extract_mfcc(audio_bytes)

                features_scaled = scaler.transform(features)

                prediction = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0][1] * 100

                st.subheader("📊 Result:")
                st.metric("Parkinson's Risk (%)", f"{prob:.2f}%")

                if prediction == 1:
                    st.error("🔴 High Risk — Similar to Parkinson's voice patterns.")
                else:
                    st.success("🟢 Low Risk — Voice does not match Parkinson's patterns.")

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("""
---
⚠ **NOTE:** This is a **screening tool**, *not* a medical diagnostic.  
Consult a doctor for medical evaluation.
""")

if __name__ == "__main__":
    main()
