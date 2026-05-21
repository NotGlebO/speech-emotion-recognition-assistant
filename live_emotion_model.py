import json
import queue
import warnings
from collections import Counter
from pathlib import Path

import joblib
import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import scipy.signal as signal
import sounddevice as sd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"

MODEL_PATH = RESULTS_DIR / "emotion_model.pkl"
ENCODER_PATH = RESULTS_DIR / "emotion_label_encoder.pkl"
FEATURE_COLUMNS_PATH = RESULTS_DIR / "feature_columns.json"
EMOTION_PROFILES_PATH = RESULTS_DIR / "class_metric_profiles.json"
MODEL_META_PATH = RESULTS_DIR / "model_meta.json"

SR = 16000
CHANNELS = 1
BLOCK_DURATION = 0.25
CALIBRATION_SECONDS = 2.0
SILENCE_SECONDS = 2.0
START_MARGIN = 2.0
SILENCE_MARGIN = 1.0
MIN_RECORDING_SECONDS = 3.0

WINDOW_SEC = 3.0
OVERLAP_SEC = 1.5
MIN_SEGMENT_SEC = 3.0
N_MFCC = 13


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
    if y is None or len(y) == 0:
        return np.array([], dtype=np.float32)

    y = y.astype(np.float32)

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
        if len(y_trimmed) > 0:
            y = y_trimmed
    except Exception:
        pass

    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0:
        y = y / peak

    return y.astype(np.float32)


def pad_if_needed(y: np.ndarray, target_length: int) -> np.ndarray:
    if y.size >= target_length:
        return y
    return np.pad(y, (0, target_length - y.size), mode="constant")


def split_audio(y: np.ndarray, sr: int, window_sec: float = WINDOW_SEC, overlap_sec: float = OVERLAP_SEC):
    window_size = int(window_sec * sr)
    step_size = int((window_sec - overlap_sec) * sr)
    min_size = int(MIN_SEGMENT_SEC * sr)

    if window_size <= 0 or step_size <= 0:
        raise ValueError("Invalid segment configuration.")

    if len(y) == 0:
        return []

    segments = []
    start = 0

    while start + window_size <= len(y):
        segment = y[start:start + window_size]

        if len(segment) >= min_size:
            segments.append(segment)

        start += step_size

    # если запись короче 3 секунд или после цикла ничего нет
    if not segments and len(y) > 0:
        segments.append(pad_if_needed(y, window_size))

    return segments

def extract_pitch_features(y: np.ndarray, sr: int):
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=70,
            fmax=350,
            sr=sr
        )
    except Exception:
        f0 = None

    if f0 is None or len(f0) == 0:
        return np.array([], dtype=float), 0.0

    f0_valid = f0[np.isfinite(f0)]
    f0_valid = f0_valid[(f0_valid >= 70) & (f0_valid <= 350)]

    voiced_ratio = float(len(f0_valid) / len(f0)) if len(f0) > 0 else 0.0

    return f0_valid, voiced_ratio


def extract_feature_dict_from_signal(y: np.ndarray, sr: int) -> dict:
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    f0_valid, voiced_ratio = extract_pitch_features(y, sr)
    pitch_range_semitones = hz_to_semitone_range(f0_valid)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = {
        "duration_sec": float(len(y) / sr),
        "voiced_ratio": voiced_ratio,
        "pitch_range_semitones": pitch_range_semitones,
    }

    # aligned names with train
    features.update(safe_stats(f0_valid, "f0"))
    features.update(safe_stats(rms, "rms"))
    features.update(safe_stats(zcr, "zcr"))
    features.update(safe_stats(centroid, "centroid"))
    features.update(safe_stats(bandwidth, "bandwidth"))
    features.update(safe_stats(rolloff, "rolloff"))
    features.update(safe_stats(flatness, "flatness"))

    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta_{i+1}_std"] = float(np.std(mfcc_delta[i]))
        features[f"mfcc_delta2_{i+1}_mean"] = float(np.mean(mfcc_delta2[i]))
        features[f"mfcc_delta2_{i+1}_std"] = float(np.std(mfcc_delta2[i]))

    return features


def rms_level(block: np.ndarray) -> float:
    block = np.asarray(block, dtype=np.float32)
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block ** 2)))


def record_until_silence():
    q = queue.Queue()
    recorded_blocks = []
    speech_started = False
    silence_time = 0.0
    total_recording_time = 0.0
    blocksize = int(SR * BLOCK_DURATION)
    calibration_levels = []

    def callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        q.put(indata.copy())

    print("Stay silent for 2 seconds. Calibrating noise floor...")

    with sd.InputStream(
        samplerate=SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        
        calibration_blocks = int(CALIBRATION_SECONDS / BLOCK_DURATION)

        for _ in range(calibration_blocks):
            block = q.get()
            mono = block[:, 0]
            calibration_levels.append(rms_level(mono))

        noise_floor = float(np.median(calibration_levels)) if calibration_levels else 0.001
        start_threshold = max(noise_floor * START_MARGIN, 0.01)
        silence_threshold = max(noise_floor * SILENCE_MARGIN, 0.006)

        print(f"Noise floor: {noise_floor:.6f}")
        print(f"Start threshold: {start_threshold:.6f}")
        print(f"Silence threshold: {silence_threshold:.6f}")
        print("Speak now. Recording stops after silence.")

        while True:
            block = q.get()
            mono = block[:, 0]
            level = rms_level(mono)

            if not speech_started:
                if level >= start_threshold:
                    speech_started = True
                    recorded_blocks.append(mono)
                    total_recording_time += BLOCK_DURATION
                    silence_time = 0.0
                    print("Speech detected. Recording...")
                continue

            recorded_blocks.append(mono)
            total_recording_time += BLOCK_DURATION

            if level < silence_threshold:
                silence_time += BLOCK_DURATION
            else:
                silence_time = 0.0

            if total_recording_time >= MIN_RECORDING_SECONDS and silence_time >= SILENCE_SECONDS:
                print("Silence detected. Analyzing...")
                break

    if not recorded_blocks:
        raise RuntimeError("No speech detected")

    audio = np.concatenate(recorded_blocks)
    return audio.astype(np.float32)


def aggregate_predictions(model, X: pd.DataFrame, encoder=None):
    probs = model.predict_proba(X)
    raw_classes = list(model.classes_)

    if encoder is not None:
        classes = [encoder.inverse_transform([c])[0] for c in raw_classes]
    else:
        classes = raw_classes

    pred_indices = [int(np.argmax(row)) for row in probs]
    pred_labels = [classes[i] for i in pred_indices]
    vote_counts = Counter(pred_labels)
    avg_probs = probs.mean(axis=0)

    best_idx = int(np.argmax(avg_probs))
    final_emotion = classes[best_idx]

    return {
        "final_emotion": final_emotion,
        "classes": classes,
        "avg_probs": avg_probs,
        "vote_counts": vote_counts,
        "segment_predictions": pred_labels,
        "segment_probabilities": probs,
    }


def build_explanation(current_metrics: dict, profiles: dict, final_emotion: str):
    metric_keys = [
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

    print("\nNumeric explanation:")
    print(f"Detected emotion profile: {final_emotion}")

    if final_emotion not in profiles:
        print("No saved emotion profile found.")
        return

    target = profiles[final_emotion]

    for key in metric_keys:
        if key not in current_metrics or key not in target:
            continue

        cur = current_metrics[key]
        ref = target[key]

        if isinstance(ref, dict):
            mean = ref.get("mean", 0.0)
            std = ref.get("std", 0.0)
        else:
            mean = float(ref)
            std = 0.0

        if std > 1e-8:
            z = (cur - mean) / std
            z_text = f"{z:+.2f}"
        else:
            z_text = "n/a"

        print(
            f"{key}: current={cur:.4f} | "
            f"{final_emotion}_mean={mean:.4f} | "
            f"diff={cur - mean:+.4f} | z={z_text}"
        )


def print_segment_timeline(result: dict, segment_sec: float = WINDOW_SEC, overlap_sec: float = OVERLAP_SEC):
    print("\nSegment timeline:")
    step = segment_sec - overlap_sec
    for i, (label, row) in enumerate(zip(result["segment_predictions"], result["segment_probabilities"])):
        start = i * step
        end = start + segment_sec
        best_prob = float(np.max(row))
        print(f"[{start:.1f}-{end:.1f}s] {label} ({best_prob:.4f})")



def predict_emotion_from_audio(audio: np.ndarray, sample_rate: int = SR, debug: bool = False) -> dict:
    """Predict emotion from an already recorded audio signal.

    This function is intended for integration with the voice assistant pipeline.
    It reuses the same preprocessing, segmentation and feature extraction logic
    as the standalone live_emotion_model.py script, but returns a structured dict
    instead of only printing results.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {ENCODER_PATH}")
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature columns not found: {FEATURE_COLUMNS_PATH}")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    audio = np.asarray(audio, dtype=np.float32).flatten()

    # If the wakeword recorder uses another sample rate, resample to SER sample rate.
    if sample_rate != SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR).astype(np.float32)

    audio = preprocess_audio(audio, SR)
    segments = split_audio(audio, SR)

    if not segments and len(audio) > 0:
        segments = [pad_if_needed(audio, int(WINDOW_SEC * SR))]

    if not segments:
        raise RuntimeError("Recorded audio is too short after preprocessing")

    feature_rows = []
    for seg in segments:
        feature_rows.append(extract_feature_dict_from_signal(seg, SR))

    X = pd.DataFrame(feature_rows).reindex(columns=feature_columns, fill_value=0.0)
    result = aggregate_predictions(model, X, encoder)

    probabilities = {
        str(emotion): float(prob)
        for emotion, prob in zip(result["classes"], result["avg_probs"])
    }

    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

    output = {
        "final_emotion": str(result["final_emotion"]),
        "probabilities": probabilities,
        "ranked_emotions": ranked,
        "segment_predictions": [str(x) for x in result["segment_predictions"]],
        "segment_count": len(segments),
    }

    if debug:
        print("\nSpeech emotion recognition result:")
        for emotion, prob in ranked:
            print(f"{emotion}: {prob:.4f}")
        print(f"Final emotion: {output['final_emotion']}")

    return output


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {ENCODER_PATH}")
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature columns not found: {FEATURE_COLUMNS_PATH}")
    if not EMOTION_PROFILES_PATH.exists():
        raise FileNotFoundError(f"Emotion profiles not found: {EMOTION_PROFILES_PATH}")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    with open(EMOTION_PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    audio = record_until_silence()
    audio = preprocess_audio(audio, SR)
    segments = split_audio(audio, SR)

    print(f"\nSegments created: {len(segments)}")
    print(f"Segment length: {WINDOW_SEC} sec")
    print(f"Overlap: {OVERLAP_SEC} sec")

    if not segments and len(audio) > 0:
        segments = [pad_if_needed(audio, int(WINDOW_SEC * SR))]

    if not segments:
        raise RuntimeError("Recorded audio is too short after preprocessing")

    feature_rows = []
    metric_rows = []

    for idx, seg in enumerate(segments):

        feats = extract_feature_dict_from_signal(seg, SR)

        print(f"\nSegment {idx}")

        print("rms_mean:", feats.get("rms_mean"))
        print("zcr_mean:", feats.get("zcr_mean"))
        print("centroid_mean:", feats.get("centroid_mean"))
        print("f0_mean:", feats.get("f0_mean"))
        print("bandwidth_mean:", feats.get("bandwidth_mean"))
        print("rolloff_mean:", feats.get("rolloff_mean"))

        feature_rows.append(feats)

        metric_rows.append({
            "duration_sec": feats.get("duration_sec", 0.0),
            "voiced_ratio": feats.get("voiced_ratio", 0.0),
            "f0_mean_hz": feats.get("f0_mean", 0.0),
            "f0_std_hz": feats.get("f0_std", 0.0),
            "pitch_range_semitones": feats.get("pitch_range_semitones", 0.0),
            "rms_mean": feats.get("rms_mean", 0.0),
            "zcr_mean": feats.get("zcr_mean", 0.0),
            "centroid_mean": feats.get("centroid_mean", 0.0),
            "bandwidth_mean": feats.get("bandwidth_mean", 0.0),
            "rolloff_mean": feats.get("rolloff_mean", 0.0),
            "flatness_mean": feats.get("flatness_mean", 0.0),
        })

    X = pd.DataFrame(feature_rows).reindex(columns=feature_columns, fill_value=0.0)
    

    result = aggregate_predictions(model, X, encoder)
    current_metrics = pd.DataFrame(metric_rows).mean(axis=0).to_dict()

    print("\nDetected metrics:")
    for k, v in current_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nSegment votes:")
    for emotion, count in result["vote_counts"].most_common():
        print(f"{emotion}: {count}")

    print("\nAverage class probabilities:")
    ranked = sorted(zip(result["classes"], result["avg_probs"]), key=lambda x: x[1], reverse=True)
    for emotion, prob in ranked:
        print(f"{emotion}: {prob:.4f}")

    print("\nFinal emotion:")
    print(result["final_emotion"])

    print_segment_timeline(result)
    build_explanation(current_metrics, profiles, result["final_emotion"])


if __name__ == "__main__":
    main()