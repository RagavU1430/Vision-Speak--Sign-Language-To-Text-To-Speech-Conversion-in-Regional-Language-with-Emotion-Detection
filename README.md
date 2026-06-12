# 🚀 VisionSpeak
### AI-Powered Multilingual Sign Language Communication & Emotion Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/MediaPipe-Hand_Tracking-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/MLP-Classification-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Supabase-Database-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/DeepFace-Emotion_AI-red?style=for-the-badge">
</p>

---

## 🌟 Overview

VisionSpeak is an intelligent real-time communication system designed to bridge the communication gap between hearing-impaired individuals and the general public.

The system recognizes hand gestures using AI, converts them into meaningful text, translates them into multiple languages, analyzes facial emotions, generates speech output, and stores interaction history in the cloud.

---

## 🎯 Problem Statement

Millions of hearing and speech-impaired individuals face communication challenges in everyday life.

VisionSpeak aims to provide:

✅ Real-Time Sign Recognition

✅ Emotion-Aware Communication

✅ Multilingual Translation

✅ Speech Generation

✅ Emergency Assistance Support

✅ Cloud-Based Interaction History

---

# 🏗️ System Architecture

```text
Webcam
   │
   ├── MediaPipe Hand Tracking
   │           │
   │           ▼
   │      Hand Landmarks
   │           │
   │           ▼
   │       MLP Model
   │           │
   │           ▼
   │    Sign Recognition
   │
   └── DeepFace
               │
               ▼
      Emotion Detection
               │
               ▼

      Combined Intelligence
               │
               ▼

      English/Tamil Translation
               │
               ▼

         Text To Speech
               │
               ▼

      Supabase Cloud Storage
```

---

# ✨ Key Features

## 🤟 Real-Time Sign Language Recognition

- Live webcam processing
- MediaPipe hand landmark detection
- MLP-based gesture classification
- High-speed inference
- Temporal stabilization

---

## 😀 Facial Emotion Detection

Using DeepFace:

- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fear
- 😐 Neutral

Emotion is displayed alongside recognized signs.

Example:

```text
Recognized Text:
HELP

Emotion:
😢 Sad
```

---

## 🌐 Multilingual Translation

Supported Languages:

- 🇬🇧 English
- 🇮🇳 Tamil

Example:

```text
English:
HELLO

Tamil:
வணக்கம்
```

---

## 🔊 Intelligent Text-To-Speech

Converts recognized signs into speech.

Supports:

- English Voice Output
- Tamil Voice Output
- Real-Time Playback

---

## 🚨 Emergency Assistance Mode

Emergency keywords are automatically detected.

Examples:

```text
HELP
EMERGENCY
DOCTOR
HOSPITAL
```

System Response:

🔴 Emergency Alert

📢 Priority Speech

📝 Event Logging

---

## ☁️ Supabase Cloud Integration

Stores:

- Recognized Text
- Translated Text
- Language Used
- Confidence Score
- Emotion
- Timestamp

Provides complete communication history.

---

# 🧠 AI Models Used

## Hand Gesture Recognition

Model:

```text
Multi-Layer Perceptron (MLP)
```

Input:

```text
21 Hand Landmarks
63 Features
```

Output:

```text
A-Z Sign Predictions
```

---

## Emotion Recognition

Model:

```text
DeepFace
```

Output:

```text
Happy
Sad
Angry
Fear
Neutral
```

---

# ⚙️ Technology Stack

## Frontend

```text
OpenCV GUI
Python Interface
```

## Backend

```text
Python
```

## AI/ML

```text
MediaPipe
Scikit-Learn
DeepFace
NumPy
```

## Database

```text
Supabase
```

## Translation

```text
GoogleTrans
```

## Speech

```text
pyttsx3
gTTS
```

---

# 📂 Project Structure

```text
VisionSpeak/
│
├── models/
│   ├── sign_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── dataset/
│   ├── A/
│   ├── B/
│   ├── ...
│   └── Z/
│
├── predict_live.py
├── train_mlp.py
├── emotion_detection.py
├── translator.py
├── speech_engine.py
├── supabase_manager.py
│
└── README.md
```

---

# 📈 Performance

| Metric | Value |
|----------|---------|
| Recognition FPS | 25-30 FPS |
| Emotion Detection | Real-Time |
| Translation Speed | < 1 sec |
| Speech Response | Instant |
| Database Logging | Live |

---

# 🎓 Academic Contribution

VisionSpeak combines:

- Computer Vision
- Machine Learning
- Emotion AI
- Natural Language Processing
- Speech Synthesis
- Cloud Computing

into a single assistive communication platform.

---

# 🔮 Future Enhancements

- Mobile Application
- WhatsApp Alert System
- Emergency SMS Notifications
- Multi-Hand Recognition
- Sentence Prediction
- Transformer-Based Models
- Regional Language Expansion
- Caregiver Dashboard

---

# 👨‍💻 Developed By

**Ragav**
Artificial Intelligence & Data Science Student

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and support accessible communication through AI.

---

<p align="center">
  <b>VisionSpeak</b><br>
  Breaking Communication Barriers Through Artificial Intelligence 🤟❤️
</p>
