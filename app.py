import streamlit as st
from audio_recorder_streamlit import audio_recorder
import joblib
import numpy as np
import soundfile as sf
import librosa
from feature import extract_parkinson_features
import io


# CONFIG

MODEL_PATH = "parkinsons_model_22.pkl"
st.set_page_config(page_title="Parkinson's Voice Detection", layout="centered")


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        st.error("❌ Model not found. Train and save 'parkinsons_model_22.pkl' first.")
        return None

model = load_model()
if model is None:
    st.stop()


# APP UI

st.title("🎙️ Parkinson’s Voice Screening")
st.write("""
Hold **'Aaaaaah'** for 3–4 seconds and record your voice.  
⚠ **Note:** This is a **screening tool**, not a medical diagnosis.
""")


# AUDIO RECORDING

audio_bytes = audio_recorder(
    text="🎤 Click to Record",
    sample_rate=16000
)

def preprocess_audio(audio_bytes):
    """Trim silence and normalize amplitude"""
    audio_buffer = io.BytesIO(audio_bytes)  # ✅ io is available now
    y, sr = sf.read(audio_buffer)

    # Mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y)

    # Normalize amplitude
    y = librosa.util.normalize(y)

    # Convert back to bytes for feature extraction
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    return buffer.getvalue()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    if st.button("🔍 Analyze Voice"):
        try:
            st.info("Processing audio...")
            processed_bytes = preprocess_audio(audio_bytes)

            # Extract features
            features = extract_parkinson_features(processed_bytes)

            # Predict probability
            prob = model.predict_proba(features)[0][1] * 100

            # Display results with probability-based risk
            st.subheader("📊 Result")
            st.metric("Parkinson's Probability (%)", f"{prob:.2f}%")

            if prob < 30:
                st.success(f"🟢 Low Risk — Probability: {prob:.2f}%")
            elif prob < 70:
                st.warning(f"🟡 Moderate Risk — Probability: {prob:.2f}%")
            else:
                st.error(f"🔴 High Risk — Probability: {prob:.2f}%")

        except Exception as e:
            st.error(f"Error analyzing voice: {e}")

# FOOTER

st.markdown("""
---
⚠ **Disclaimer:** This application is for educational/screening purposes only.  
Consult a qualified physician for medical evaluation.
""")
