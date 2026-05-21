import json
import warnings
from pathlib import Path

import joblib
import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import scipy.signal as signal

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

MODEL_PATH = BASE_DIR / "emotion_model.pkl"
ENCODER_PATH = BASE_DIR / "emotion_label_encoder.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.json"
CLASS_PROFILES_PATH = BASE_DIR / "class_metric_profiles.json"
MODEL_META_PATH = BASE_DIR / "model_meta.json"

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3
SEGMENT_OVERLAP_SECONDS = 1.5
N_MFCC = 13
RANDOM_STATE = 42


def safe_stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
    }


def hz_to_semitone_range(f0: np.ndarray) -> float:
    f0 = np.asarray(f0, dtype=float)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]

    if f0.size < 2:
        return 0.0

    return float(12.0 * np.log2(np.max(f0) / np.min(f0)))


def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    if y is None or y.size == 0:
        return np.array([], dtype=np.float32)

    try:
        y = nr.reduce_noise(y=y, sr=sr, stationary=False)
    except Exception:
        pass

    try:
        sos = signal.butter(4, 80, btype="highpass", fs=sr, output="sos")
        y = signal.sosfilt(sos, y)
    except Exception:
        pass

    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if y_trimmed.size > 0:
            y = y_trimmed
    except Exception:
        pass

    max_abs = np.max(np.abs(y)) if y.size > 0 else 0.0
    if max_abs > 0:
        y = y / max_abs

    return y.astype(np.float32)


def pad_if_needed(y: np.ndarray, target_length: int) -> np.ndarray:
    if y.size >= target_length:
        return y
    return np.pad(y, (0, target_length - y.size), mode="constant")


def split_audio_into_segments(
    y: np.ndarray,
    sr: int,
    segment_seconds: float = SEGMENT_SECONDS,
    overlap_seconds: float = SEGMENT_OVERLAP_SECONDS,
) -> list[np.ndarray]:
    segment_length = int(segment_seconds * sr)
    step = int((segment_seconds - overlap_seconds) * sr)

    if segment_length <= 0 or step <= 0:
        raise ValueError("Invalid segment configuration.")

    if y.size == 0:
        return []

    if y.size <= segment_length:
        return [pad_if_needed(y, segment_length)]

    segments = []
    start = 0

    while start + segment_length <= y.size:
        segments.append(y[start:start + segment_length])
        start += step

    if start < y.size:
        tail = y[-segment_length:]
        segments.append(pad_if_needed(tail, segment_length))

    return segments


def extract_segment_features(y: np.ndarray, sr: int) -> tuple[dict, dict]:
    duration_sec = float(y.size / sr) if sr > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    f0, _, _ = librosa.pyin(
        y,
        fmin=70,
        fmax=350,
        sr=sr,
    )
    f0_valid = f0[np.isfinite(f0)] if f0 is not None else np.array([])
    f0_valid = f0_valid[(f0_valid >= 70) & (f0_valid <= 350)]
    voiced_ratio = float(len(f0_valid) / len(f0)) if f0 is not None and f0.size > 0 else 0.0
    pitch_range_semitones = hz_to_semitone_range(f0_valid)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = {
        "duration_sec": duration_sec,
        "voiced_ratio": voiced_ratio,
        "pitch_range_semitones": pitch_range_semitones,
    }

    features.update(safe_stats(f0_valid, "f0"))
    features.update(safe_stats(rms, "rms"))
    features.update(safe_stats(zcr, "zcr"))
    features.update(safe_stats(centroid, "centroid"))
    features.update(safe_stats(bandwidth, "bandwidth"))
    features.update(safe_stats(rolloff, "rolloff"))
    features.update(safe_stats(flatness, "flatness"))

    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta_{i+1}_std"] = float(np.std(mfcc_delta[i]))

        features[f"mfcc_delta2_{i+1}_mean"] = float(np.mean(mfcc_delta2[i]))
        features[f"mfcc_delta2_{i+1}_std"] = float(np.std(mfcc_delta2[i]))

    report_metrics = {
        "duration_sec": round(duration_sec, 3),
        "voiced_ratio": round(voiced_ratio, 3),
        "f0_mean_hz": round(features["f0_mean"], 2),
        "f0_std_hz": round(features["f0_std"], 2),
        "pitch_range_semitones": round(pitch_range_semitones, 2),
        "rms_mean": round(features["rms_mean"], 6),
        "zcr_mean": round(features["zcr_mean"], 6),
        "centroid_mean": round(features["centroid_mean"], 2),
        "bandwidth_mean": round(features["bandwidth_mean"], 2),
        "rolloff_mean": round(features["rolloff_mean"], 2),
        "flatness_mean": round(features["flatness_mean"], 6),
    }

    return features, report_metrics


def load_audio_segments(audio_path: Path) -> list[np.ndarray]:
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    y = preprocess_audio(y, sr)
    return split_audio_into_segments(y, sr)


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rows = []
    labels = []
    metric_rows = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    emotion_dirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    if not emotion_dirs:
        raise FileNotFoundError("No emotion folders found inside dataset directory.")

    for emotion_dir in emotion_dirs:
        emotion_label = emotion_dir.name

        for audio_path in emotion_dir.rglob("*"):
            if not audio_path.is_file() or audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                segments = load_audio_segments(audio_path)
                if not segments:
                    continue

                for idx, segment in enumerate(segments, start=1):
                    features, metrics = extract_segment_features(segment, SAMPLE_RATE)

                    rows.append(features)
                    labels.append(emotion_label)

                    metric_row = {
                        "label": emotion_label,
                        "source_file": audio_path.name,
                        "segment_index": idx,
                    }
                    metric_row.update(metrics)
                    metric_rows.append(metric_row)

                print(f"[OK] {audio_path.name} -> {emotion_label} -> {len(segments)} segments")

            except Exception as exc:
                print(f"[ERROR] {audio_path}: {exc}")

    if not rows:
        raise ValueError("No valid audio segments were extracted from the dataset.")

    X = pd.DataFrame(rows).fillna(0.0)
    y = np.array(labels)
    metrics_df = pd.DataFrame(metric_rows)

    return X, y, metrics_df


def build_class_profiles(metrics_df: pd.DataFrame) -> dict:
    metric_columns = [
        "duration_sec",
        "voiced_ratio",
        "f0_mean_hz",
        "f0_std_hz",
        "pitch_range_semitones",
        "rms_mean",
        "zcr_mean",
        "centroid_mean",
        "bandwidth_mean",
        "rolloff_mean",
        "flatness_mean",
    ]

    grouped = metrics_df.groupby("label")[metric_columns].mean().round(6)
    return grouped.to_dict(orient="index")


def main():
    print(f"[INFO] Dataset directory: {DATASET_DIR}")
    X, y_text, metrics_df = load_dataset(DATASET_DIR)


    
    
    print(f"[INFO] Segments: {len(X)}")
    print(f"[INFO] Classes: {sorted(set(y_text))}")
    print(f"[INFO] Features: {X.shape[1]}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ))
    ])

    
    

    

    model.fit(X, y)

    class_profiles = build_class_profiles(metrics_df)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    with open(CLASS_PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_profiles, f, indent=2)

    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_rate": SAMPLE_RATE,
                "segment_seconds": SEGMENT_SECONDS,
                "segment_overlap_seconds": SEGMENT_OVERLAP_SECONDS,
                "n_mfcc": N_MFCC,
                "classes": label_encoder.classes_.tolist(),
                "model_type": "SVM_RBF",
                "evaluation_type": "train_test_split_stratified",
            },
            f,
            indent=2,
        )

    print(f"[INFO] Model saved to: {MODEL_PATH}")
    print(f"[INFO] Encoder saved to: {ENCODER_PATH}")
    print(f"[INFO] Feature columns saved to: {FEATURE_COLUMNS_PATH}")
    print(f"[INFO] Class metric profiles saved to: {CLASS_PROFILES_PATH}")
    print(f"[INFO] Model meta saved to: {MODEL_META_PATH}")


if __name__ == "__main__":
    main()