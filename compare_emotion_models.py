import os
import pickle
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dd")
RESULTS_DIR = os.path.join(BASE_DIR, "model_comparison_results")
FEATURES_CACHE_PATH = os.path.join(RESULTS_DIR, "features.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)



def extract_features(file_path: str) -> np.ndarray | None:
    try:
        signal, sr = librosa.load(file_path, sr=None, duration=3.0)

        # Basic MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Delta
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta_mfcc, axis=1)
        delta_std = np.std(delta_mfcc, axis=1)

        # Delta-delta
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        delta2_mean = np.mean(delta2_mfcc, axis=1)
        delta2_std = np.std(delta2_mfcc, axis=1)

        # Extra simple audio features
        zcr = librosa.feature.zero_crossing_rate(signal)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        rms = librosa.feature.rms(y=signal)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)

        features = np.hstack([
            mfcc_mean, mfcc_std,
            delta_mean, delta_std,
            delta2_mean, delta2_std,
            [
                zcr_mean, zcr_std,
                rms_mean, rms_std,
                spectral_centroid_mean, spectral_centroid_std
            ]
        ])

        return features.astype(np.float32)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None



def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    labels = ["angry", "happy", "sad", "neutral"]

    print(f"Dataset path: {path}")
    print(f"Dataset exists: {os.path.exists(path)}")

    for label in labels:
        folder = os.path.join(path, label)

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
        print(f"Found {len(files)} files in {label}")

        for file_name in files:
            file_path = os.path.join(folder, file_name)
            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded dataset: X shape = {X.shape}, y shape = {y.shape}")
    return X, y



def load_or_create_features() -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(FEATURES_CACHE_PATH):
        print("Loading cached features...")
        with open(FEATURES_CACHE_PATH, "rb") as f:
            X, y = pickle.load(f)
        print(f"Cached features loaded: X shape = {X.shape}, y shape = {y.shape}")
        return X, y

    print("Extracting features from audio files...")
    X, y = load_dataset(DATASET_PATH)

    with open(FEATURES_CACHE_PATH, "wb") as f:
        pickle.dump((X, y), f)

    print(f"Features saved to {FEATURES_CACHE_PATH}")
    return X, y

def save_confusion_matrix_image(cm: np.ndarray, class_names: np.ndarray, model_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(
    rf"$\it{{{model_name}}}$ kļūdu matrica")
    ax.set_xlabel("Prognozētā klase")
    ax.set_ylabel("Patiesā klase")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_metrics_image(model_name: str, accuracy: float, f1_macro: float, out_path: str):
    metrics = ["Precizitāte", "F1 rādītājs (makro)"]
    values = [accuracy, f1_macro]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(metrics, values)

    ax.set_ylim(0, 1)
    ax.set_title(
        rf"$\it{{{model_name}}}$ veiktspējas rādītāji")
    ax.set_ylabel("Vērtība")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def evaluate_models_train_test(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    latvian_labels = {
        "angry": "Dusmas",
        "happy": "Prieks",
        "neutral": "Neitrāls",
        "sad": "Skumjas"
    }

    class_names_lv = [latvian_labels[c] for c in class_names]

    print("Classes:", list(class_names))



    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print(f"Dataset shape: {X.shape}")
    print(f"Cross-validation folds: 5")

    # Models
    models = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance"
            ))
        ]),

        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ])
    }

    results = []

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)
        model_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(model_dir, exist_ok=True)

        # Cross-validation prediction
        y_pred = cross_val_predict(
            model,
            X,
            y_encoded,
            cv=cv,
            n_jobs=-1
        )

        # Metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        f1_macro = f1_score(y_encoded, y_pred, average="macro")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")

        report = classification_report(
            y_encoded,
            y_pred,
            target_names=class_names,
            digits=4
        )
        print("\nClassification report:")
        print(report)

        cm = confusion_matrix(y_encoded, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        print("\nKļūdu matrica:")
        print(cm_df)

       # Save report
        report_path = os.path.join(model_dir, f"{name}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Save confusion matrix CSV
        cm_path = os.path.join(model_dir, f"{name}_confusion_matrix.csv")
        cm_df.to_csv(cm_path, encoding="utf-8-sig")

        # Save confusion matrix image
        cm_img_path = os.path.join(model_dir, f"{name}_confusion_matrix.png")
        save_confusion_matrix_image(cm, class_names_lv, name, cm_img_path)

        # Save accuracy / F1 image
        metrics_img_path = os.path.join(model_dir, f"{name}_metrics.png")
        save_metrics_image(name, accuracy, f1_macro, metrics_img_path)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "F1_macro": f1_macro
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1_macro", ascending=False).reset_index(drop=True)

    results_csv_path = os.path.join(RESULTS_DIR, "results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {results_csv_path}")
    return results_df



if __name__ == "__main__":
    X, y = load_or_create_features()

    if len(X) == 0 or len(y) == 0:
        print("No data loaded. Check dataset path and folder structure.")
    else:
        evaluate_models_train_test(X, y)
