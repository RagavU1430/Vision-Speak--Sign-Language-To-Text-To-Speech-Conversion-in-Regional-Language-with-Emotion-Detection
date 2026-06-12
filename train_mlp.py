"""
Sign Language A-Z Recognition — MLP Training Pipeline
======================================================
Single-file pipeline that:
  1. Extracts 21 hand landmarks (63 features) from images using MediaPipe
  2. Trains a scikit-learn MLPClassifier
  3. Outputs a classification report and saves a styled confusion matrix
"""

# ── Suppress all warnings BEFORE any other imports ──────────────────────────
import os, warnings

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── Standard / third-party imports ──────────────────────────────────────────
import csv
import string
import joblib
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from utils import extract_enhanced_features  # Enhanced landmarks and engineered features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI needed
import matplotlib.pyplot as plt

# ── Constants ───────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join("archive", "asl_alphabet_train", "asl_alphabet_train")
CSV_PATH = "extracted_landmarks.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
LABELS = list(string.ascii_uppercase)  # A-Z only
LIMIT_SAMPLES_PER_GESTURE = 200  # Set to None for full training (3000/class). Set to a number (e.g. 200) for faster training.


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Extract Landmarks
# ═════════════════════════════════════════════════════════════════════════════
def extract_landmarks() -> str:
    """
    Walk through each A-Z subfolder, run MediaPipe Hands on every image,
    and write the 63-feature vector + label to a CSV file.
    Returns the path to the saved CSV.
    """
    print("\n" + "=" * 60)
    print("  STEP 1 - Extracting Hand Landmarks with MediaPipe")
    print("=" * 60)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    # Build CSV header for 99 features: f0, f1, ..., f98, label
    header = [f"f{i}" for i in range(99)]
    header.append("label")

    total_images = 0
    processed = 0
    skipped = 0

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label in sorted(LABELS):
            folder = os.path.join(DATASET_DIR, label)
            if not os.path.isdir(folder):
                print(f"  [WARN] Folder not found for label '{label}', skipping.")
                continue

            files = [
                fn
                for fn in os.listdir(folder)
                if fn.lower().endswith(VALID_EXTENSIONS)
            ]
            if LIMIT_SAMPLES_PER_GESTURE is not None:
                files = sorted(files)[:LIMIT_SAMPLES_PER_GESTURE]
            total_images += len(files)

            for fn in tqdm(files, desc=f"  {label}", unit="img", leave=False):
                img_path = os.path.join(folder, fn)
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    # Extract 99-dimensional enhanced features
                    row = list(extract_enhanced_features(hand))
                    row.append(label)
                    writer.writerow(row)
                    processed += 1
                else:
                    skipped += 1

    hands.close()

    print(f"\n  Total Images    : {total_images}")
    print(f"  Processed (OK)  : {processed}")
    print(f"  Skipped (no hand): {skipped}")
    print(f"  Saved to        : {CSV_PATH}")
    return CSV_PATH


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Train MLP
# ═════════════════════════════════════════════════════════════════════════════
def augment_landmarks_and_recompute(coords):
    """
    Applies data augmentation (Rotation, Scaling, Noise) to 3D hand coordinates,
    then re-extracts the 99 enhanced features.
    """
    # 1. Rotation: Rotate around Z-axis (in-plane rotation) by a random angle [-15, 15] degrees
    angle = np.random.uniform(-15, 15) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    rotated = np.dot(coords, rot_mat.T)
    
    # 2. Scaling: slight hand size variation [0.95, 1.05]
    scale = np.random.uniform(0.95, 1.05)
    scaled = rotated * scale
    
    # 3. Noise injection: simulate tracking jitter
    noise = np.random.normal(0, 0.005, size=coords.shape).astype(np.float32)
    jittered = scaled + noise
    
    # Re-extract engineered features from augmented coordinates
    return extract_enhanced_features(jittered)


def augment_dataset(X_raw, y_raw, factor=3):
    """
    Augments the raw feature dataset.
    For each sample, we keep the original and add `factor` augmented copies.
    """
    X_aug = []
    y_aug = []
    
    for i in range(len(X_raw)):
        sample = X_raw[i]
        label = y_raw[i]
        
        # Keep original
        X_aug.append(sample)
        y_aug.append(label)
        
        # The first 63 features are the 21 normalized landmarks (x, y, z)
        coords = sample[:63].reshape(21, 3)
        
        # Generate augmented copies
        for _ in range(factor):
            aug_feat = augment_landmarks_and_recompute(coords)
            X_aug.append(aug_feat)
            y_aug.append(label)
            
    return np.array(X_aug, dtype=np.float32), np.array(y_aug)


def train_mlp(csv_path: str):
    """
    Load the landmark CSV, augment features, scale features, train an MLPClassifier,
    and persist the model + encoder + scaler to disk.
    Returns (model, label_encoder, scaler, X_test, y_test).
    """
    print("\n" + "=" * 60)
    print("  STEP 2 - Training MLP Classifier with Data Augmentation")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────────────────────
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"\n  Samples loaded  : {len(df)}")
    print(f"  Unique labels   : {sorted(df['label'].unique())}")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    # ── Encode labels ───────────────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ── Train / test split on RAW features first ────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Raw train samples: {len(X_train_raw)}")
    print(f"  Raw test samples : {len(X_test_raw)}")

    # ── Apply Data Augmentation to the training set only ───────────────────
    print("\n  Augmenting training set (rotation, scaling, noise injection)...")
    X_train_aug, y_train_aug = augment_dataset(X_train_raw, y_train, factor=3)
    print(f"  Augmented train samples: {len(X_train_aug)}")

    # ── Scale features ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_aug)
    X_test = scaler.transform(X_test_raw)
    y_train = y_train_aug  # use augmented labels for training

    # ── Build & fit MLP ─────────────────────────────────────────────────────
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
        random_state=42,
    )

    print("\n  Training started …\n")
    mlp.fit(X_train, y_train)

    # ── Save artefacts ──────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(mlp, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n  Model saved     : {MODEL_PATH}")
    print(f"  Encoder saved   : {ENCODER_PATH}")
    print(f"  Scaler saved    : {SCALER_PATH}")

    return mlp, le, scaler, X_test, y_test


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Report & Confusion Matrix
# ═════════════════════════════════════════════════════════════════════════════
def generate_report(model, label_encoder, X_test, y_test):
    """
    Print the classification report, compute accuracy,
    and save a styled confusion-matrix PNG.
    Returns the test accuracy as a float.
    """
    print("\n" + "=" * 60)
    print("  STEP 3 - Evaluation Report & Confusion Matrix")
    print("=" * 60)

    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_

    # ── Classification report ───────────────────────────────────────────────
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(f"\n{report}")

    accuracy = accuracy_score(y_test, y_pred) * 100.0
    print(f"  Test Accuracy   : {accuracy:.2f}%")

    # ── Confusion matrix plot ───────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Teal-based colormap
    cmap = plt.cm.BuGn

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=9, color="white")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9, color="white")

    ax.set_xlabel("Predicted Label", fontsize=13, color="white", labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, color="white", labelpad=10)
    ax.set_title(
        "Sign Language MLP - Confusion Matrix",
        fontsize=16,
        color="white",
        pad=18,
        fontweight="bold",
    )

    ax.tick_params(axis="both", colors="white")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Confusion matrix saved -> {CM_PATH}")

    return accuracy


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Step 1
    csv_file = extract_landmarks()

    # Step 2
    model, le, scaler, X_test, y_test = train_mlp(csv_file)

    # Step 3
    acc = generate_report(model, le, X_test, y_test)

    # ── Final summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 30)
    print("  === TRAINING COMPLETE ===")
    print("=" * 30)
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Encoder   : {ENCODER_PATH}")
    print(f"  Scaler    : {SCALER_PATH}")
    print(f"  Confusion : {CM_PATH}")
    print(f"  Accuracy  : {acc:.2f}%")
    print("=" * 30)
