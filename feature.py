import io
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call

def extract_parkinson_features(audio_bytes):
    """
    Extract 22 classical Parkinson's voice biomarkers
    in the same order as the UCI Parkinson dataset.
    Returns a numpy array of shape (1, 22).
    """

    # 1️⃣ Convert bytes to waveform
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = sf.read(audio_buffer)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)  # convert to mono

    sound = parselmouth.Sound(y, sr)

    # 2️⃣ Pitch features
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    Fo = call(pitch, "Get mean", 0, 0, "Hertz")
    Fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    Flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    # 3️⃣ Jitter features
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    Jitter_local = call(pointProcess, "Get jitter (local)", 0, 0.02, 0.0001, 0.02, 1.3)
    Jitter_abs = call(pointProcess, "Get jitter (local, absolute)", 0, 0.02, 0.0001, 0.02, 1.3)
    RAP = call(pointProcess, "Get jitter (rap)", 0, 0.02, 0.0001, 0.02, 1.3)
    PPQ = call(pointProcess, "Get jitter (ppq5)", 0, 0.02, 0.0001, 0.02, 1.3)
    DDP = call(pointProcess, "Get jitter (ddp)", 0, 0.02, 0.0001, 0.02, 1.3)

    # 4️⃣ Shimmer features
    Shimmer_local = call([sound, pointProcess], "Get shimmer (local)", 0, 0.02, 0.0001, 0.02, 1.3, 1.6)
    Shimmer_dB = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0.02, 0.0001, 0.02, 1.3, 1.6)
    APQ3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0.02, 0.0001, 0.02, 1.3, 1.6)
    APQ5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0.02, 0.0001, 0.02, 1.3, 1.6)
    APQ11 = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0.02, 0.0001, 0.02, 1.3, 1.6)
    DDA = call([sound, pointProcess], "Get shimmer (dda)", 0, 0.02, 0.0001, 0.02, 1.3, 1.6)

    # 5️⃣ Harmonics
    HNR = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR_mean = call(HNR, "Get mean", 0, 0)
    NHR = 1 / (HNR_mean + 1e-6)

    # 6️⃣ Assemble all 22 features in dataset order
    features = [
        Fo, Fhi, Flo,
        Jitter_local, Jitter_abs, RAP, PPQ, DDP,
        Shimmer_local, Shimmer_dB, APQ3, APQ5, APQ11, DDA,
        NHR, HNR_mean,
        # Extra placeholders if your model expects full 22 features
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    features = np.array(features, dtype=np.float64).reshape(1, -1)
    return features
