# 🤟 VisionSpeak — Sign Language to Speech Conversion

> **Real-time ASL Recognition** using MediaPipe hand tracking + MLP classifier with Text-to-Speech output in regional language and emotion detection.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

---

## 📌 Project Title

**VisionSpeak** — *Sign Language to Text-to-Speech Conversion in Regional Language with Emotion Detection*

---

## 📋 Requirement Gathering

### Functional Requirements

| #  | Requirement |
|----|-------------|
| FR1 | Capture live webcam feed and detect hand landmarks in real time |
| FR2 | Recognize ASL alphabets A–Z from hand gestures |
| FR3 | Build words automatically from consecutively detected letters |
| FR4 | Convert recognized text to speech in a regional language |
| FR5 | Collect and manage a labeled dataset of 2000 images per gesture |
| FR6 | Extract 21 MediaPipe hand landmarks (63 features) from each image |
| FR7 | Train and persist an MLP model, label encoder, and feature scaler |
| FR8 | Output a confusion matrix and classification report after training |

### Non-Functional Requirements

| #  | Requirement |
|----|-------------|
| NFR1 | Real-time inference with stable FPS (≥ 20 fps on standard hardware) |
| NFR2 | Landmark extraction must be reproducible across sessions |
| NFR3 | Landmark CSV must survive partial runs (append-safe structure) |
| NFR4 | Model artefacts stored in a dedicated `/models` directory |
| NFR5 | All MediaPipe / TensorFlow C++ logs suppressed for clean UX |
| NFR6 | Skeleton canvas must be normalized to 400×400 for consistent input |

---

## 🎯 Objective Definition

### Primary Objective
Build an **end-to-end, real-time Sign Language Recognition (SLR)** pipeline that bridges the communication gap for the hearing-impaired by converting hand gestures into audible speech.

### Sub-Objectives

```
1. Dataset Collection    →  Capture skeleton-canvas images per gesture (A–Z)
2. Landmark Extraction   →  Convert images to 63-float MediaPipe feature vectors
3. Model Training        →  Train a 3-layer MLP (512 → 256 → 128) on landmarks
4. Real-Time Prediction  →  Live webcam inference with debounce & word-building
5. Speech Output         →  TTS synthesis in regional language
```

### Success Metrics
- Test accuracy ≥ 90% across all 26 ASL classes
- Confusion matrix saved as a visual artefact for analysis
- Live prediction latency < 50 ms per frame

---

## 👤 User & Module Identification

### Users

| Role | Description |
|------|-------------|
| **End User** | Hearing-impaired individual who signs in front of the webcam |
| **Caregiver / Audience** | Receives the speech output to understand the signer |
| **ML Developer** | Runs `train_mlp.py` to re-train on new datasets |
| **Data Collector** | Uses `collect_dataset.py` to build/expand the gesture dataset |

### Modules

```
visionspeak/
├── collect_dataset.py   →  MODULE 1: Webcam-based A–Z gesture dataset collector
├── hand_detection.py    →  MODULE 2: Real-time hand landmark detector + HUD
├── utils.py             →  MODULE 3: SkeletonGenerator, stderr suppressor
├── train_mlp.py         →  MODULE 4: Full training pipeline (extract → train → report)
├── predict.py           →  MODULE 5: Static image inference
├── predict_live.py      →  MODULE 6: Live webcam inference + TTS
└── test_load.py         →  MODULE 7: Model integrity check
```

---

## 📊 Use Case Diagram

```
                         ┌─────────────────────────────────────────┐
                         │              VisionSpeak System          │
                         │                                         │
  ┌──────────────┐       │  ┌─────────────┐   ┌────────────────┐  │
  │  Data        │──────▶│  │ Collect     │   │ Hand Detection │  │
  │  Collector   │       │  │ Dataset     │   │   (Live HUD)   │  │
  └──────────────┘       │  └──────┬──────┘   └───────┬────────┘  │
                         │         │                   │           │
  ┌──────────────┐       │  ┌──────▼──────┐            │           │
  │  ML          │──────▶│  │  Train MLP  │            │           │
  │  Developer   │       │  │  Pipeline   │            │           │
  └──────────────┘       │  └──────┬──────┘            │           │
                         │         │                   │           │
  ┌──────────────┐       │  ┌──────▼──────┐   ┌────────▼───────┐  │
  │  End User    │──────▶│  │  Live       │──▶│  TTS Output    │  │
  │  (Signer)    │       │  │  Predict    │   │  (Regional)    │  │
  └──────────────┘       │  └─────────────┘   └────────────────┘  │
                         │                                         │
  ┌──────────────┐       │  ┌─────────────────────────────────┐   │
  │  Caregiver   │◀──────│  │        Speech / Text Output      │   │
  │  Audience    │       │  └─────────────────────────────────┘   │
  └──────────────┘       └─────────────────────────────────────────┘
```

**Use Cases:**
- `UC1` — Collect gesture images (A–Z) via webcam
- `UC2` — Extract MediaPipe landmarks to CSV
- `UC3` — Train MLP classifier on landmark data
- `UC4` — Evaluate model (report + confusion matrix)
- `UC5` — Predict sign language from live webcam
- `UC6` — Convert recognized text to speech

---

## 🗄️ Database Requirement Analysis

VisionSpeak uses a **file-based data layer** (no RDBMS), structured as follows:

### Dataset Store (`/archive/`)
- One subdirectory per class label (`A/` through `Z/`)
- Each directory holds up to **2,000 PNG skeleton-canvas images** (400×400 px)
- Total target: **52,000 images** (26 classes × 2,000)

### Landmark CSV (`extracted_landmarks.csv`)
- **Rows:** One per successfully detected hand
- **Columns:** 63 feature floats (`x1,y1,z1` … `x21,y21,z21`) + `label`
- **Approximate size at full dataset:** ~60,000 rows

```
x1,y1,z1, x2,y2,z2, ..., x21,y21,z21, label
0.457, 0.583, -6.47e-7, ..., 0.403, 0.457, -0.028, A
...
```

### Model Artefacts (`/models/`)

| File | Content | Library |
|------|---------|---------|
| `mlp_model.pkl` | Trained `MLPClassifier` | scikit-learn / joblib |
| `label_encoder.pkl` | `LabelEncoder` (A–Z mapping) | scikit-learn / joblib |
| `scaler.pkl` | `StandardScaler` (fit on train split) | scikit-learn / joblib |
| `confusion_matrix.png` | 26×26 styled confusion matrix | matplotlib |

### Data Flow Summary

```
Webcam Frames
     │
     ▼
MediaPipe Hands  ──▶  21 Landmarks (x, y, z)
     │
     ▼
SkeletonGenerator  ──▶  400×400 PNG  ──▶  /archive/<LABEL>/
     │
     ▼
extract_landmarks()  ──▶  extracted_landmarks.csv
     │
     ▼
StandardScaler  ──▶  train_test_split (80/20)
     │
     ▼
MLPClassifier (512→256→128)  ──▶  mlp_model.pkl
     │
     ▼
classification_report + confusion_matrix.png
```

---

## 🧮 ER Diagram Design

Since the system uses **flat-file storage**, the logical entity model maps as follows:

```
┌─────────────────────┐         ┌──────────────────────────┐
│       GESTURE        │         │         SAMPLE            │
├─────────────────────┤  1───N  ├──────────────────────────┤
│ PK  label  CHAR(1)  │────────▶│ PK  sample_id  INT        │
│     target_count INT│         │ FK  label      CHAR(1)    │
│     actual_count INT│         │     image_path VARCHAR    │
└─────────────────────┘         │     timestamp  BIGINT     │
                                 └──────────────────────────┘
                                           │
                                           │ extracted by
                                           ▼
                                 ┌──────────────────────────┐
                                 │        LANDMARK_ROW       │
                                 ├──────────────────────────┤
                                 │ PK  row_id     INT        │
                                 │ FK  label      CHAR(1)    │
                                 │     x1–x21    FLOAT[21]   │
                                 │     y1–y21    FLOAT[21]   │
                                 │     z1–z21    FLOAT[21]   │
                                 └────────────┬─────────────┘
                                              │ trains
                                              ▼
                                 ┌──────────────────────────┐
                                 │         MLP_MODEL         │
                                 ├──────────────────────────┤
                                 │ PK  model_id   INT        │
                                 │     hidden_layers TEXT    │
                                 │     activation  VARCHAR   │
                                 │     accuracy    FLOAT     │
                                 │     saved_path  VARCHAR   │
                                 │     created_at  DATETIME  │
                                 └──────────────────────────┘
```

**Key Relationships:**
- `GESTURE` 1 → N `SAMPLE` — each gesture label has many image samples
- `SAMPLE` 1 → 1 `LANDMARK_ROW` — each image yields one 63-float feature vector
- `LANDMARK_ROW` N → 1 `MLP_MODEL` — all rows collectively train one model

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install mediapipe opencv-python scikit-learn matplotlib tqdm joblib pandas

# 2. Collect dataset (A–Z gestures)
python collect_dataset.py

# 3. Train the MLP model
python train_mlp.py

# 4. Run live prediction
python predict_live.py
```

---

## 📁 Project Structure

```
VisionSpeak/
├── archive/                    # Gesture image dataset (A–Z folders)
├── models/                     # Trained model artefacts
│   ├── mlp_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── confusion_matrix.png
├── collect_dataset.py          # Dataset collection via webcam
├── hand_detection.py           # Hand detection + HUD demo
├── train_mlp.py                # Full MLP training pipeline
├── predict.py                  # Static image prediction
├── predict_live.py             # Live webcam prediction + TTS
├── test_load.py                # Model integrity verification
├── utils.py                    # Shared utilities
├── extracted_landmarks.csv     # Landmark feature vectors
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Hand Tracking | MediaPipe Hands (21 landmarks) |
| Computer Vision | OpenCV |
| ML Model | scikit-learn MLPClassifier |
| Feature Engineering | StandardScaler, LabelEncoder |
| Visualization | Matplotlib |
| Speech Output | pyttsx3 / gTTS |
| Language | Python 3.10+ |

---

## 📜 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
