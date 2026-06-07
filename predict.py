"""
================================================================================
Sign Language Predictor v5.0 — Production-Grade Pipeline
================================================================================

Pipeline Architecture:
  Webcam → Lighting Normalization → MediaPipe Hands → EMA Smoothing →
  ROI Crop → Skeleton Canvas → CNN Inference → Stability Voting →
  Gesture Lock → Word Builder → HUD Display

Feature Summary:
  1. Mode Toggle          — [P] Performance (model_complexity=0, higher FPS)
                            [A] Accuracy   (model_complexity=1, better precision)
  2. Frame Skipping       — Process prediction on every 2nd frame, display every frame
  3. EMA Landmark Smooth  — smoothed = 0.7 * previous + 0.3 * current
  4. Tracking Recovery    — 0.5s grace period before resetting on hand loss
  5. Adaptive ROI         — Crop hand region to reduce MediaPipe search area
  6. Lighting Robustness  — Conditional CLAHE when mean brightness is low
  7. Motion Stability     — Pause predictions during rapid hand transitions
  8. Averaged FPS         — Rolling window average over last 30 frame intervals
  9. Debug Panel          — Hand Detected, Tracking Quality, Landmarks Visible,
                            Confidence, Stability %, Tracking Status, FPS
 10. Enhanced Drawing     — thickness=3, circle_radius=4 for clear landmark display
 11. Auto-Capture         — Hold letter 1.5s to add to word buffer
 12. Gesture Locking      — Lock stable prediction for 1s to prevent flicker

Controls:
  [P]     → Performance Mode (model_complexity=0)
  [A]     → Accuracy Mode (model_complexity=1)
  [SPACE] → Manually add letter to word buffer
  [C]     → Clear word buffer
  [ESC]   → Quit

Dependencies: Python, OpenCV, MediaPipe, TensorFlow, NumPy (standard stack)
================================================================================
"""

import os
import sys
import time
import math
import pickle
import contextlib
import warnings
import threading
import logging
from collections import deque, Counter

# ==================== SUPPRESS TF/MP C++ NOISE ====================
# Must be set before any TensorFlow or MediaPipe import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

# Suppress Python logging noise from TF and absl
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from utils import suppress_c_stderr, SkeletonGenerator


with suppress_c_stderr():
    import cv2
    import numpy as np
    import tensorflow as tf
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


# ==================== CONFIGURATION CONSTANTS ====================

# File paths (relative to script directory)
# Try enhanced model first, fall back to original
MODEL_PATH_ENHANCED = "models/sign_cnn_enhanced.keras"
MODEL_PATH_ORIGINAL = "models/sign_cnn.h5"
LABEL_ENCODER_PATH_ENHANCED = "models/label_encoder_enhanced.pkl"
LABEL_ENCODER_PATH_ORIGINAL = "models/label_encoder.pkl"
METADATA_PATH = "models/model_metadata.pkl"

# Default image size - will be updated from metadata if available
IMAGE_SIZE = (128, 128)  # Default to enhanced model size

# CNN prediction thresholds
CONFIDENCE_THRESHOLD = 0.90    # Below this → "UNKNOWN"
STABILITY_THRESHOLD = 0.80     # 80% majority agreement in prediction queue
QUEUE_SIZE = 20                # Temporal smoothing window (last N frames)

# Auto-capture timing
HOLD_TIME = 1.5                # Seconds to hold a letter before auto-adding

# Auto-speak timing — speak buffer automatically when hand absent this long
AUTO_SPEAK_DELAY = 2.0         # Seconds of no hand before auto-speaking

# EMA smoothing weights — higher previous weight = smoother but more lag
EMA_WEIGHT_PREV = 0.7         # Weight for previous smoothed landmark
EMA_WEIGHT_CURR = 0.3         # Weight for current raw landmark (must sum to 1.0)

# Tracking recovery — how long to keep last landmarks after hand disappears
TRACKING_GRACE_PERIOD = 0.5   # Seconds before resetting tracking state

# Motion detection — average landmark displacement threshold (normalized coords)
MOTION_THRESHOLD = 0.015      # Above this → "TRACKING..." pause

# Frame skipping — process CNN every Nth frame to save compute
FRAME_SKIP_INTERVAL = 2       # Predict every 2nd frame, display every frame

# Lighting normalization — CLAHE applied only when frame is dark
BRIGHTNESS_THRESHOLD = 80     # Mean brightness below this triggers CLAHE
CLAHE_CLIP_LIMIT = 2.0        # Contrast limit for CLAHE tiles
CLAHE_TILE_SIZE = (8, 8)      # Grid size for CLAHE

# FPS averaging window
FPS_HISTORY_SIZE = 30          # Average over last 30 frame timestamps

# MediaPipe mode presets — toggled at runtime with [P] and [A] keys
MODE_PERFORMANCE = {
    "name": "PERFORMANCE",
    # model_complexity=0: Lightweight palm detection model. ~2x faster than
    # complexity=1 but slightly less accurate for unusual hand orientations.
    # Best for real-time use where FPS matters more than edge-case accuracy.
    "model_complexity": 0,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.7,
}
MODE_ACCURACY = {
    "name": "ACCURACY",
    # model_complexity=1: Full palm detection model with additional layers.
    # ~15-20% slower but handles rotated/partially-occluded hands better.
    # Use when prediction accuracy is paramount and hardware can sustain 25+ FPS.
    "model_complexity": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.7,
}


# ==================== CLASS: HandTracker ====================

class HandTracker:
    """
    Manages MediaPipe Hands lifecycle, EMA landmark smoothing, motion velocity
    measurement, tracking recovery with grace period, and runtime mode switching.

    Key behaviors:
      - smooth_landmarks(): Applies EMA formula (0.7 * prev + 0.3 * curr) to all
        21 landmarks independently, reducing high-frequency jitter while keeping
        responsiveness for intentional movements.
      - calculate_motion(): Measures average Euclidean displacement between current
        and previous frame landmarks to detect rapid hand transitions.
      - Grace period: When hand detection drops, keeps last known landmarks for
        0.5 seconds before resetting. Prevents one-frame detection gaps from
        causing flickering resets in the stability pipeline.
      - switch_mode(): Destroys current MediaPipe instance and recreates with
        new model_complexity, allowing real-time Performance/Accuracy toggling.
    """

    def __init__(self, mode_config=None):
        self.mp_hands_module = mp.solutions.hands
        self.mode_config = mode_config or MODE_PERFORMANCE
        self.hands = self._create_hands(self.mode_config)

        # EMA smoothing state
        self.prev_landmarks = None  # List of 21 (x, y, z) tuples after smoothing

        # Tracking recovery state
        self.last_seen_time = 0     # time.time() when hand was last detected
        self.cached_landmarks = None  # Last known smoothed landmarks for grace period
        self.landmarks_visible = 0   # Count of landmarks with valid visibility
        self.prev_crop_box = None   # Box: (x1, y1, x2, y2) in pixel space

    def _create_hands(self, config):
        """Instantiate mp.solutions.hands.Hands with given config dict."""
        return self.mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=config["model_complexity"],
            min_detection_confidence=config["min_detection_confidence"],
            min_tracking_confidence=config["min_tracking_confidence"],
        )

    def switch_mode(self, new_config):
        """
        Hot-swap MediaPipe model complexity at runtime.
        Closes old instance, creates new one, resets smoothing state.
        Called when user presses [P] or [A].
        """
        if new_config["name"] == self.mode_config["name"]:
            return  # Already in this mode

        self.hands.close()
        self.mode_config = new_config
        self.hands = self._create_hands(new_config)
        self.reset()
        print(f"[MODE] Switched to {new_config['name']} mode "
              f"(model_complexity={new_config['model_complexity']})")

    def smooth_landmarks(self, current_landmarks):
        """
        Exponential Moving Average across all 21 hand landmarks.

        Formula per coordinate:
          smoothed = 0.7 * previous_smoothed + 0.3 * current_raw

        The 0.7 weight on previous values acts as a low-pass filter:
          - Dampens frame-to-frame jitter (high-frequency noise from MediaPipe)
          - Preserves intentional movements (low-frequency signal)
          - 0.3 current weight ensures responsiveness within 3-5 frames

        Returns list of 21 smoothed (x, y, z) tuples.
        """
        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current_landmarks):
            # First frame or landmark count changed — initialize without smoothing
            self.prev_landmarks = list(current_landmarks)
            return list(current_landmarks)

        smoothed = []
        for curr, prev in zip(current_landmarks, self.prev_landmarks):
            sx = EMA_WEIGHT_PREV * prev[0] + EMA_WEIGHT_CURR * curr[0]
            sy = EMA_WEIGHT_PREV * prev[1] + EMA_WEIGHT_CURR * curr[1]
            sz = EMA_WEIGHT_PREV * prev[2] + EMA_WEIGHT_CURR * curr[2]
            smoothed.append((sx, sy, sz))

        self.prev_landmarks = smoothed
        return smoothed

    def calculate_motion(self, current_landmarks):
        """
        Average Euclidean distance between current and previous landmark positions.
        Used to detect rapid hand movements and pause prediction updates.
        Values are in normalized coordinates (0.0–1.0 range).
        Typical stable hand: 0.001–0.005. Rapid transition: 0.02+.
        """
        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current_landmarks):
            return 0.0

        total_dist = 0.0
        for curr, prev in zip(current_landmarks, self.prev_landmarks):
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            dz = curr[2] - prev[2]
            total_dist += math.sqrt(dx * dx + dy * dy + dz * dz)

        return total_dist / len(current_landmarks)

    def process_frame(self, rgb_frame):
        """
        Run MediaPipe hand detection on the RGB frame.
        Uses adaptive ROI cropping if available.
        Returns (hand_landmarks_object, landmarks_visible_count) or (None, 0).
        Updates tracking recovery timestamps.
        """
        frame_h, frame_w = rgb_frame.shape[:2]
        
        # Determine if we should crop the input frame
        cropped = False
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, 0, 0
        
        if self.prev_crop_box is not None:
            crop_x1, crop_y1, crop_x2, crop_y2 = self.prev_crop_box
            # Ensure crop box is within frame boundaries and not degenerate
            if crop_x2 - crop_x1 >= 50 and crop_y2 - crop_y1 >= 50:
                crop_img = rgb_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                cropped = True
                
        # Run inference
        if cropped:
            results = self.hands.process(crop_img)
        else:
            results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            
            # Map landmarks back to full-frame space if cropped
            if cropped:
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                
                mapped_lms = landmark_pb2.NormalizedLandmarkList()
                for lm in hand_lms.landmark:
                    new_lm = mapped_lms.landmark.add()
                    new_lm.x = (crop_x1 + lm.x * crop_w) / frame_w
                    new_lm.y = (crop_y1 + lm.y * crop_h) / frame_h
                    new_lm.z = lm.z * crop_w / frame_w  # Scale depth proportionally
                    if hasattr(lm, 'visibility'):
                        new_lm.visibility = lm.visibility
                hand_lms = mapped_lms

            self.landmarks_visible = sum(
                1 for lm in hand_lms.landmark
                if not hasattr(lm, 'visibility') or lm.visibility > 0.5
            )
            self.last_seen_time = time.time()
            return hand_lms, self.landmarks_visible

        # Fallback to full frame immediately if crop failed (reduces jitter)
        if cropped:
            results_fallback = self.hands.process(rgb_frame)
            if results_fallback.multi_hand_landmarks:
                hand_lms = results_fallback.multi_hand_landmarks[0]
                self.landmarks_visible = sum(
                    1 for lm in hand_lms.landmark
                    if not hasattr(lm, 'visibility') or lm.visibility > 0.5
                )
                self.last_seen_time = time.time()
                return hand_lms, self.landmarks_visible

        self.prev_crop_box = None
        return None, 0

    def is_within_grace_period(self):
        """
        Returns True if hand was lost less than TRACKING_GRACE_PERIOD seconds ago.
        During this window, cached landmarks are still available for display
        continuity, preventing single-frame detection drops from causing
        a full pipeline reset and visual flicker.
        """
        if self.last_seen_time == 0:
            return False
        return (time.time() - self.last_seen_time) < TRACKING_GRACE_PERIOD

    def reset(self):
        """Full reset — called when grace period expires or mode switches."""
        self.prev_landmarks = None
        self.cached_landmarks = None
        self.landmarks_visible = 0
        self.prev_crop_box = None

    def close(self):
        self.hands.close()


# SkeletonGenerator class is imported from utils.py


# ==================== CLASS: PredictionEngine ====================

class PredictionEngine:
    """
    Loads and runs the CNN sign language classifier.
    Supports both original (64x64) and enhanced (128x128) models.
    Automatically detects input size from model metadata or model architecture.
    """

    def __init__(self, model_path, encoder_path, metadata_path=None):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.metadata_path = metadata_path
        self.model = None
        self.label_encoder = None
        self.input_size = (128, 128)  # Default to enhanced model size

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {os.path.abspath(self.model_path)}. Train the CNN first."
            )
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(
                f"Encoder not found: {os.path.abspath(self.encoder_path)}. Train the CNN first."
            )

        print("Loading models... please wait.")
        with suppress_c_stderr():
            self.model = tf.keras.models.load_model(self.model_path)
        with open(self.encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Try to load metadata for input size
        if self.metadata_path and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.input_size = tuple(metadata.get('image_size', (128, 128)))
                print(f"Loaded model metadata. Input size: {self.input_size}")
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                # Fallback: infer from model input shape
                self._infer_input_size()
        else:
            # Fallback: infer from model input shape
            self._infer_input_size()

        print(f"Models loaded successfully. Input size: {self.input_size}")

    def _infer_input_size(self):
        """Infer input size from model architecture."""
        try:
            if hasattr(self.model, 'input_shape'):
                shape = self.model.input_shape
                if len(shape) >= 3:
                    self.input_size = (shape[1], shape[2])
                    print(f"Inferred input size from model: {self.input_size}")
                    return
        except Exception:
            pass
        # Default fallback
        self.input_size = (128, 128)
        print(f"Using default input size: {self.input_size}")

    def predict(self, skeleton_img):
        """
        Run CNN inference on a skeleton canvas image.
        Automatically resizes to the model's expected input size.
        Returns (predicted_label: str, confidence: float, top_3_predictions: list).
        """
        resized = cv2.resize(skeleton_img, self.input_size)
        normalized = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # Direct functional evaluation for faster real-time inference
        preds = self.model(input_data, training=False).numpy()[0]
        class_idx = np.argmax(preds)
        confidence = float(preds[class_idx])
        label = str(self.label_encoder.inverse_transform([class_idx])[0])

        # Get top 3 predictions
        top_3_indices = np.argsort(preds)[::-1][:3]
        top_3 = []
        for idx in top_3_indices:
            lbl = str(self.label_encoder.inverse_transform([idx])[0])
            conf = float(preds[idx])
            top_3.append((lbl, conf))

        return label, confidence, top_3


# ==================== CLASS: StabilityManager ====================

class StabilityManager:
    """
    Temporal prediction smoothing with majority voting, gesture locking,
    and motion-based prediction pausing.

    Pipeline per frame:
      1. Check motion velocity → if above threshold, emit "TRACKING..." and pause
      2. Append raw prediction to sliding window (deque of QUEUE_SIZE)
      3. Calculate majority vote and stability percentage
      4. If gesture is locked (within lock_duration), maintain locked prediction
         unless confidence drops below threshold
      5. If stability >= 80%, lock the prediction for 1 second
      6. Otherwise, emit "UNSTABLE"
    """

    def __init__(self, queue_size=QUEUE_SIZE, stability_threshold=STABILITY_THRESHOLD,
                 lock_duration=1.0, motion_threshold=MOTION_THRESHOLD):
        self.queue_size = queue_size
        self.stability_threshold = stability_threshold
        self.lock_duration = lock_duration
        self.motion_threshold = motion_threshold
        self.history = deque(maxlen=queue_size)

        # Gesture lock state — prevents flicker for 1 second after stable prediction
        self.locked_letter = None
        self.lock_start_time = 0

    def is_locked(self):
        """Check if a gesture lock is currently active and not expired."""
        if self.locked_letter is not None:
            if time.time() - self.lock_start_time < self.lock_duration:
                return True
            else:
                self.locked_letter = None
        return False

    def update(self, raw_pred, confidence, hand_motion):
        """
        Process one frame's raw prediction through the stability pipeline.

        Returns: (stable_prediction: str, stability_pct: float, tracking_status: str)
          - stable_prediction: The smoothed output letter or status string
          - stability_pct: 0.0–1.0 agreement ratio in the prediction queue
          - tracking_status: "LOCKED" | "STABLE" | "UNSTABLE" | "TRACKING..."
        """
        # 1. Motion gate — pause predictions during rapid hand movement
        if hand_motion > self.motion_threshold:
            self.locked_letter = None
            self.history.append("UNSTABLE")
            return "TRACKING...", 0.0, "TRACKING..."

        # 2. Temporal queue — add current prediction
        self.history.append(raw_pred)

        if not self.history:
            return "UNKNOWN", 0.0, "STABLE"

        # 3. Majority voting
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        stability_pct = float(count) / len(self.history)

        # 4. Gesture lock — maintain stable prediction for lock_duration
        if self.is_locked():
            if confidence < CONFIDENCE_THRESHOLD or raw_pred == "UNKNOWN":
                self.locked_letter = None  # Break lock on confidence drop
            else:
                return self.locked_letter, stability_pct, "LOCKED"

        # 5. Stability check — requires 80% majority of valid predictions
        invalid_preds = {"UNKNOWN", "UNSTABLE", "TRACKING...", None}
        if stability_pct >= self.stability_threshold and most_common not in invalid_preds:
            self.locked_letter = most_common
            self.lock_start_time = time.time()
            return most_common, stability_pct, "LOCKED"
        elif stability_pct < self.stability_threshold:
            return "UNSTABLE", stability_pct, "UNSTABLE"
        else:
            return most_common, stability_pct, "STABLE"

    def reset(self):
        self.history.clear()
        self.locked_letter = None


# ==================== CLASS: WordBuilder ====================

class WordBuilder:
    """
    Manages the output word buffer with automatic hands-free letter capture.

    Auto-capture flow:
      1. Track which letter is currently being held (effective_letter)
      2. Start timer when a stable letter first appears
      3. Reset timer whenever the letter changes
      4. After HOLD_TIME seconds of the same letter → add to buffer
      5. Prevent duplicate consecutive additions
      6. Require reset gesture (hand removal or UNKNOWN) before next capture
      7. Flash confirmation on screen for 1 second after capture
    """

    def __init__(self, hold_time=HOLD_TIME):
        self.hold_time = hold_time
        self.word_buffer = ""
        self.current_letter = None
        self.start_time = 0
        self.last_added = ""
        self.letter_confirmed = False
        self.confirm_flash = None
        self.confirm_flash_time = 0
        self.capture_status = "WAITING"

    def update(self, stable_pred, hand_detected):
        """Called every frame to update auto-capture state machine."""
        # Expire confirmation flash after 1 second
        if self.confirm_flash and (time.time() - self.confirm_flash_time > 1.0):
            self.confirm_flash = None

        # Hand lost → reset lock to allow next letter capture
        if not hand_detected:
            if self.letter_confirmed:
                self.letter_confirmed = False
                self.capture_status = "WAITING"
            self.current_letter = None
            self.start_time = time.time()
            return

        # Extract valid letter (filter out status strings and 'nothing' neutral pose)
        invalid_preds = {"UNKNOWN", "UNSTABLE", "TRACKING...", "nothing", None}
        effective_letter = stable_pred if stable_pred not in invalid_preds else None

        # Detect letter change → reset timer
        if effective_letter != self.current_letter:
            self.current_letter = effective_letter
            self.start_time = time.time()
            if effective_letter is None and self.letter_confirmed:
                self.letter_confirmed = False
                self.capture_status = "WAITING"

        # Auto-capture hold timing
        if self.current_letter is not None and not self.letter_confirmed:
            elapsed = time.time() - self.start_time
            self.capture_status = "CAPTURING"

            if elapsed >= self.hold_time:
                if self.current_letter != self.last_added:
                    if self.current_letter == "del":
                        self.backspace()
                        self.last_added = "del"
                        self.letter_confirmed = True
                        self.capture_status = "ADDED"
                        self.confirm_flash = "DEL"
                        self.confirm_flash_time = time.time()
                    elif self.current_letter == "space":
                        self.word_buffer += " "
                        self.last_added = "space"
                        self.letter_confirmed = True
                        self.capture_status = "ADDED"
                        self.confirm_flash = "SPACE"
                        self.confirm_flash_time = time.time()
                        print(f"[AUTO-CAPTURE] Added Space → buffer: {self.word_buffer}")
                    else:
                        self.word_buffer += self.current_letter
                        self.last_added = self.current_letter
                        self.letter_confirmed = True
                        self.capture_status = "ADDED"
                        self.confirm_flash = self.current_letter
                        self.confirm_flash_time = time.time()
                        print(f"[AUTO-CAPTURE] Added '{self.current_letter}' → buffer: {self.word_buffer}")
                else:
                    self.letter_confirmed = True
                    self.capture_status = "ADDED"
        elif self.letter_confirmed:
            self.capture_status = "ADDED"
        else:
            self.capture_status = "WAITING"

    def add_manual(self, stable_pred):
        """Manual letter addition via [SPACE] key."""
        invalid_preds = {"UNKNOWN", "UNSTABLE", "TRACKING...", "nothing", None}
        if stable_pred not in invalid_preds:
            if stable_pred == "del":
                self.backspace()
                self.last_added = "del"
                self.letter_confirmed = True
                self.capture_status = "ADDED"
                self.confirm_flash = "DEL"
                self.confirm_flash_time = time.time()
            elif stable_pred == "space":
                self.word_buffer += " "
                self.last_added = "space"
                self.letter_confirmed = True
                self.capture_status = "ADDED"
                self.confirm_flash = "SPACE"
                self.confirm_flash_time = time.time()
                print(f"[MANUAL] Added Space → buffer: {self.word_buffer}")
            else:
                self.word_buffer += stable_pred
                self.last_added = stable_pred
                self.letter_confirmed = True
                self.capture_status = "ADDED"
                self.confirm_flash = stable_pred
                self.confirm_flash_time = time.time()
                print(f"[MANUAL] Added '{stable_pred}' → buffer: {self.word_buffer}")

    def backspace(self):
        """Delete last character from word buffer."""
        if self.word_buffer:
            self.word_buffer = self.word_buffer[:-1]
            self.last_added = self.word_buffer[-1] if self.word_buffer else ""
            self.confirm_flash = None
            self.letter_confirmed = False
            self.capture_status = "WAITING"
            print(f"[BACKSPACE] Removed last char → buffer: {self.word_buffer}")

    def clear(self):
        """Reset entire word buffer and capture state."""
        self.word_buffer = ""
        self.last_added = ""
        self.letter_confirmed = False
        self.capture_status = "WAITING"
        self.current_letter = None
        self.confirm_flash = None
        print("Cleared word buffer.")

    def get_elapsed_time(self):
        if self.current_letter is not None and not self.letter_confirmed:
            return time.time() - self.start_time
        elif self.letter_confirmed:
            return self.hold_time
        return 0.0


# ==================== TEXT TO SPEECH ====================

# Initialize TTS engine once at module startup (not per-call)
USE_GTTS = False
_tts_engine = None

try:
    import pyttsx3
    with suppress_c_stderr():
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', 150)     # 150 WPM — clear, not too fast
        _tts_engine.setProperty('volume', 1.0)   # Maximum volume
        # Prefer a female voice if available, otherwise fall back to first voice
        _voices = _tts_engine.getProperty('voices')
        _female_voice = None
        for _v in _voices:
            if 'female' in _v.name.lower() or 'zira' in _v.name.lower():
                _female_voice = _v
                break
        if _female_voice:
            _tts_engine.setProperty('voice', _female_voice.id)
        elif _voices:
            _tts_engine.setProperty('voice', _voices[0].id)
    print(f"[TTS] Engine initialized: pyttsx3")
except Exception as _e:
    print(f"[TTS] pyttsx3 unavailable ({_e}), falling back to gTTS.")
    USE_GTTS = True
    _tts_engine = None


def _speak_blocking_pyttsx3(text):
    """Blocking pyttsx3 speech — runs in a daemon thread."""
    try:
        # pyttsx3 is not thread-safe; create a fresh engine per thread
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        female_voice = None
        for v in voices:
            if 'female' in v.name.lower() or 'zira' in v.name.lower():
                female_voice = v
                break
        if female_voice:
            engine.setProperty('voice', female_voice.id)
        elif voices:
            engine.setProperty('voice', voices[0].id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        print("[TTS] Done.")
    except Exception as e:
        print(f"[TTS] pyttsx3 error: {e}")


def _speak_blocking_gtts(text):
    """Blocking gTTS speech — generates MP3 via Google, plays via pygame."""
    try:
        # pyrefly: ignore [missing-import]
        from gtts import gTTS
        # pyrefly: ignore [missing-import]
        import pygame
        import tempfile

        # Generate speech audio file
        tts = gTTS(text=text, lang='en')
        tmp_path = os.path.join(tempfile.gettempdir(), "sign_lang_tts.mp3")
        tts.save(tmp_path)

        # Play using pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        print("[TTS] Done.")
    except Exception as e:
        print(f"[TTS] gTTS error: {e}")


def speak_text_async(text):
    """
    Speaks the provided text asynchronously in a daemon thread.
    Uses pyttsx3 as primary engine; falls back to gTTS if unavailable.
    Never blocks the main webcam loop.
    """
    if not text or not text.strip():
        return

    if USE_GTTS:
        print(f"[TTS] Engine: gTTS")
        target_fn = _speak_blocking_gtts
    else:
        print(f"[TTS] Engine: pyttsx3")
        target_fn = _speak_blocking_pyttsx3

    thread = threading.Thread(target=target_fn, args=(text,), daemon=True)
    thread.start()


# ==================== LIGHTING NORMALIZATION ====================

def normalize_lighting(frame):
    """
    Conditionally apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    only when the frame is dark (mean brightness < BRIGHTNESS_THRESHOLD).

    This avoids unnecessary processing in well-lit conditions while significantly
    improving hand detection in low-light environments. CLAHE is applied per-tile
    (8×8 grid) on the L channel of LAB color space to preserve color balance.

    Returns: (processed_frame, was_clahe_applied: bool)
    """
    # Convert to LAB to analyze luminance independently of color
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_brightness = np.mean(l_channel)

    if mean_brightness < BRIGHTNESS_THRESHOLD:
        # Low-light detected — apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        lab[:, :, 0] = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced, True

    # Well-lit — pass through unchanged (zero overhead)
    return frame, False


# ==================== ADAPTIVE ROI CROPPING ====================

def compute_roi(landmarks, frame_w, frame_h, margin_ratio=0.3):
    """
    Compute a Region of Interest (ROI) bounding box around detected landmarks.
    Used to crop the hand area for the next frame's MediaPipe processing,
    reducing computational load and improving detection for small/far hands.

    The margin_ratio adds padding around the tight bounding box to accommodate
    hand movement between frames without losing tracking.

    Returns: (x1, y1, x2, y2) in pixel coordinates, or None if landmarks invalid.
    """
    if not landmarks:
        return None

    xs = [pt[0] * frame_w for pt in landmarks]
    ys = [pt[1] * frame_h for pt in landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    roi_w = max_x - min_x
    roi_h = max_y - min_y

    # Expand ROI by margin_ratio on each side
    margin_x = roi_w * margin_ratio
    margin_y = roi_h * margin_ratio

    x1 = max(0, int(min_x - margin_x))
    y1 = max(0, int(min_y - margin_y))
    x2 = min(frame_w, int(max_x + margin_x))
    y2 = min(frame_h, int(max_y + margin_y))

    # Minimum ROI size to avoid degenerate crops
    if (x2 - x1) < 50 or (y2 - y1) < 50:
        return None

    return (x1, y1, x2, y2)


# ==================== FPS COUNTER ====================

class FPSCounter:
    """
    Rolling average FPS calculator over the last FPS_HISTORY_SIZE frame timestamps.
    More stable than instantaneous 1/dt which fluctuates wildly.
    """

    def __init__(self, history_size=FPS_HISTORY_SIZE):
        self.timestamps = deque(maxlen=history_size)

    def tick(self):
        """Record current timestamp. Call once per frame."""
        self.timestamps.append(time.time())

    def get_fps(self):
        """
        Calculate average FPS from timestamp history.
        Returns 0.0 if insufficient data (< 2 timestamps).
        """
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


# ==================== HUD RENDERING ====================

def determine_tracking_quality(stability_pct, hand_motion, hand_detected):
    """
    Map raw metrics to a human-readable tracking quality label.
    Used in the debug panel for at-a-glance system health assessment.
    """
    if not hand_detected:
        return "NONE", (120, 120, 120)
    if hand_motion > MOTION_THRESHOLD:
        return "POOR", (0, 0, 255)
    if stability_pct >= 0.80:
        return "GOOD", (0, 255, 0)
    elif stability_pct >= 0.50:
        return "MEDIUM", (0, 255, 255)
    else:
        return "POOR", (0, 0, 255)


def draw_cyber_corners(img, top_left, bottom_right, color, thickness=2, size=15):
    x1, y1 = top_left
    x2, y2 = bottom_right
    # Top Left
    cv2.line(img, (x1, y1), (x1 + size, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + size), color, thickness)
    # Top Right
    cv2.line(img, (x2 - 1, y1), (x2 - 1 - size, y1), color, thickness)
    cv2.line(img, (x2 - 1, y1), (x2 - 1, y1 + size), color, thickness)
    # Bottom Left
    cv2.line(img, (x1, y2 - 1), (x1 + size, y2 - 1), color, thickness)
    cv2.line(img, (x1, y2 - 1), (x1, y2 - 1 - size), color, thickness)
    # Bottom Right
    cv2.line(img, (x2 - 1, y2 - 1), (x2 - 1 - size, y2 - 1), color, thickness)
    cv2.line(img, (x2 - 1, y2 - 1), (x2 - 1, y2 - 1 - size), color, thickness)


def draw_ui(frame, canvas, raw_letter, confidence, stable_pred, stability_pct,
            word_buffer, fps, queue_len, tracking_status, hand_detected,
            capture_status, elapsed_time, confirm_flash, mode_name,
            landmarks_visible, tracking_quality, tq_color, clahe_active,
            top_3=None, speaking_flash=None):
    """
    Composes the unified premium dashboard: side-by-side webcam + dark neon skeleton grid on top,
    three-column cybernetic diagnostics panel on bottom, and segmented auto-capture hold progress.
    """
    h_target = 400
    aspect = frame.shape[1] / frame.shape[0]
    w_target = int(h_target * aspect)

    resized_feed = cv2.resize(frame, (w_target, h_target))

    # Styled displays
    display_canvas = canvas.copy()
    # Replace white canvas background with dark space background
    white_mask = (display_canvas[:, :, 0] == 255) & (display_canvas[:, :, 1] == 255) & (display_canvas[:, :, 2] == 255)
    display_canvas[white_mask] = [18, 18, 18]
    
    # Draw technical grid lines on canvas
    grid_c = (32, 32, 32)
    for x in range(0, display_canvas.shape[1], 40):
        cv2.line(display_canvas, (x, 0), (x, display_canvas.shape[0]), grid_c, 1)
    for y in range(0, display_canvas.shape[0], 40):
        cv2.line(display_canvas, (0, y), (display_canvas.shape[1], y), grid_c, 1)

    combined_w = w_target + canvas.shape[1]
    combined_h = h_target + 240

    window = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    window[0:h_target, 0:w_target] = resized_feed
    window[0:h_target, w_target:combined_w] = display_canvas

    # Cyber HUD color system (BGR)
    CYAN = (255, 230, 0)
    MAGENTA = (255, 0, 200)
    GREEN = (0, 255, 0)
    TEXT_WHITE = (245, 245, 245)
    TEXT_MUTED = (140, 140, 140)
    BORDER_GRAY = (45, 45, 45)

    # Confidence state mapping
    if confidence >= 0.95:
        status_color = GREEN
        status_label = "EVAL: OPTIMAL"
    elif confidence >= 0.90:
        status_color = (0, 220, 255) # Yellow/gold
        status_label = "EVAL: STANDARD"
    else:
        status_color = (0, 0, 255) # Red
        status_label = "EVAL: UNSTABLE"

    # 1. Overlay bottom dashboard background with panel accent lines
    overlay = window.copy()
    cv2.rectangle(overlay, (0, h_target), (combined_w, combined_h), (14, 14, 14), -1)
    
    col_w = combined_w // 3
    # Draw vertical dashboard dividers
    cv2.line(overlay, (col_w, h_target + 10), (col_w, combined_h - 15), BORDER_GRAY, 1)
    cv2.line(overlay, (2 * col_w, h_target + 10), (2 * col_w, combined_h - 15), BORDER_GRAY, 1)
    
    # Draw double cybernetic separating bar with a glowing cyan line
    cv2.line(overlay, (0, h_target), (combined_w, h_target), CYAN, 2)
    cv2.line(overlay, (0, h_target - 1), (combined_w, h_target - 1), CYAN, 1)

    # Apply translucency
    alpha = 0.95
    cv2.addWeighted(overlay, alpha, window, 1 - alpha, 0, window)

    # 2. Draw cyber corners around the video feeds
    draw_cyber_corners(window, (0, 0), (w_target, h_target), CYAN, 2, 18)
    draw_cyber_corners(window, (w_target, 0), (combined_w, h_target), GREEN, 2, 18)

    # 3. Webcam feed overlay details: pulsing REC dot & Tracking Status
    pulse = (int(time.time() * 3) % 2) == 0
    rec_c = (0, 0, 255) if pulse else (0, 0, 80)
    cv2.circle(window, (25, 25), 6, rec_c, -1)
    cv2.putText(window, "LIVE MONITOR", (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    sys_color = GREEN if hand_detected else (0, 150, 255)
    sys_text = "HAND ACTIVE" if hand_detected else "STANDBY"
    cv2.circle(window, (w_target - 110, 25), 5, sys_color, -1)
    cv2.putText(window, sys_text, (w_target - 98, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    # ---- Column 1: System Info & Diagnostics ----
    c1 = 20
    cv2.putText(window, "SYSTEM DIAGNOSTICS", (c1, h_target + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

    cv2.putText(window, f"Processing FPS:  {int(fps)}", (c1, h_target + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)

    mode_c = CYAN if mode_name == "PERFORMANCE" else (0, 200, 255)
    cv2.putText(window, f"Pipeline Mode:   {mode_name}", (c1, h_target + 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_c, 1, cv2.LINE_AA)

    # Hand presence detection status
    hand_color = GREEN if hand_detected else (0, 0, 230)
    cv2.putText(window, "Hand Status: ", (c1, h_target + 81),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.circle(window, (c1 + 95, h_target + 77), 4, hand_color, -1)
    cv2.putText(window, "CONNECTED" if hand_detected else "DISCONNECTED", (c1 + 106, h_target + 81),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, hand_color, 1, cv2.LINE_AA)

    # Landmarks count
    cv2.putText(window, f"Landmarks Rec:   {landmarks_visible}/21 Nodes", (c1, h_target + 99),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)

    # Tracking Quality pill
    cv2.putText(window, "Track Quality:   ", (c1, h_target + 117),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(window, (c1 + 105, h_target + 104), (c1 + 185, h_target + 120), tq_color, -1)
    cv2.putText(window, tracking_quality, (c1 + 115, h_target + 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # Confidence pill
    cv2.rectangle(window, (c1, h_target + 128), (c1 + 155, h_target + 144), status_color, -1)
    cv2.putText(window, status_label, (c1 + 12, h_target + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)

    if clahe_active:
        cv2.putText(window, "[CLAHE ONLINE]", (c1 + 175, h_target + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    # ---- Column 2: Live Inference ----
    c2 = col_w + 20
    cv2.putText(window, "REAL-TIME STABILITY", (c2, h_target + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

    cv2.putText(window, f"Queue Window:  {queue_len}/{QUEUE_SIZE}", (c2, h_target + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)

    stab_color = GREEN if stability_pct >= STABILITY_THRESHOLD else (0, 120, 255)
    cv2.putText(window, f"Stability index: {int(stability_pct * 100)}%", (c2, h_target + 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, stab_color, 1, cv2.LINE_AA)

    cv2.putText(window, "Voting State:  ", (c2, h_target + 81),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)
    track_colors = {
        "LOCKED": GREEN, "STABLE": (0, 200, 0),
        "UNSTABLE": (0, 255, 255), "TRACKING...": (0, 120, 255),
        "NO HAND": (100, 100, 100), "RECOVERY": MAGENTA,
    }
    tc = track_colors.get(tracking_status, (100, 100, 100))
    cv2.rectangle(window, (c2 + 105, h_target + 68), (c2 + 205, h_target + 84), tc, -1)
    cv2.putText(window, tracking_status, (c2 + 111, h_target + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # Glowing Circular Reticle for Predicted Character
    scanner_center = (col_w + 240, h_target + 95)
    scanner_radius = 42
    cv2.circle(window, scanner_center, scanner_radius, CYAN, 1)
    cv2.circle(window, scanner_center, scanner_radius - 5, (40, 40, 40), 1)
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        x1_t = int(scanner_center[0] + (scanner_radius - 8) * math.cos(rad))
        y1_t = int(scanner_center[1] + (scanner_radius - 8) * math.sin(rad))
        x2_t = int(scanner_center[0] + scanner_radius * math.cos(rad))
        y2_t = int(scanner_center[1] + scanner_radius * math.sin(rad))
        cv2.line(window, (x1_t, y1_t), (x2_t, y2_t), CYAN, 1)

    stable_char = stable_pred if stable_pred else "-"
    long_status = {"UNSTABLE", "UNKNOWN", "TRACKING...", "RECOVERY", "NO HAND", "-"}
    if stable_char in long_status:
        text_size = cv2.getTextSize(stable_char, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        tx = scanner_center[0] - text_size[0] // 2
        ty = scanner_center[1] + text_size[1] // 2
        cv2.putText(window, stable_char, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 250), 1, cv2.LINE_AA)
    else:
        text_size = cv2.getTextSize(stable_char, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
        tx = scanner_center[0] - text_size[0] // 2
        ty = scanner_center[1] + text_size[1] // 2
        cv2.putText(window, stable_char, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.8, GREEN, 4, cv2.LINE_AA)

    # ---- Column 3: Word Buffer Console & Top Predictions ----
    c3 = 2 * col_w + 20
    cv2.putText(window, "CYBERNETIC WORD SHELL", (c3, h_target + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

    cv2.rectangle(window, (c3, h_target + 35), (combined_w - 20, h_target + 82), (20, 20, 20), -1)
    cv2.rectangle(window, (c3, h_target + 35), (combined_w - 20, h_target + 82), CYAN, 1)

    caret = "_" if (int(time.time() * 2) % 2 == 0) else " "
    wt = f"> {word_buffer}{caret}" if word_buffer else f"> [READY]{caret}"
    wc = GREEN if word_buffer else TEXT_MUTED
    cv2.putText(window, wt, (c3 + 12, h_target + 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, wc, 2, cv2.LINE_AA)

    # Keyboard shortcut label
    cv2.putText(window, "[SPACE] Add  |  [C] Clear  |  [BS] Undo  |  [ENTER] Speak",
                (c3, h_target + 97),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_MUTED, 1, cv2.LINE_AA)

    # Top Predictions
    if top_3:
        for idx, (lbl, conf) in enumerate(top_3):
            y_bar = h_target + 110 + idx * 16
            cv2.putText(window, f"{lbl}:", (c3, y_bar + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(window, f"{conf*100:.0f}%", (c3 + 22, y_bar + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_MUTED, 1, cv2.LINE_AA)
            
            bar_width = 160
            bar_x = c3 + 70
            cv2.rectangle(window, (bar_x, y_bar + 2), (bar_x + bar_width, y_bar + 9), (30, 30, 30), -1)
            cv2.rectangle(window, (bar_x, y_bar + 2), (bar_x + bar_width, y_bar + 9), (50, 50, 50), 1)
            
            fill_width = int(bar_width * conf)
            if fill_width > 0:
                bar_color = (int(255 * (1 - conf)), 255, int(255 * conf))
                cv2.rectangle(window, (bar_x, y_bar + 2), (bar_x + fill_width, y_bar + 9), bar_color, -1)

    # ---- Auto-Capture Progress Bar ----
    bar_y = h_target + 175
    cv2.putText(window, f"Hold Progress: {elapsed_time:.1f}/{HOLD_TIME:.1f}s", (c1, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)

    bx1 = c1
    bx2 = col_w * 2 - 20
    bw = bx2 - bx1
    fill = min(elapsed_time / HOLD_TIME, 1.0)
    fw = int(bw * fill)

    by1, by2 = bar_y + 20, bar_y + 30
    cv2.rectangle(window, (bx1, by1), (bx2, by2), (32, 32, 32), -1)

    bc = GREEN if fill >= 1.0 else (CYAN if fill >= 0.5 else (0, 150, 255))
    if fw > 0:
        cv2.rectangle(window, (bx1, by1), (bx1 + fw, by2), bc, -1)
    cv2.rectangle(window, (bx1, by1), (bx2, by2), BORDER_GRAY, 1)

    # Buffer Status Pill
    cap_map = {
        "WAITING": ((100, 100, 100), "WAITING"),
        "CAPTURING": (CYAN, "CAPTURING..."),
        "ADDED": (GREEN, "LETTER CAPTURED"),
    }
    sc, st = cap_map.get(capture_status, ((100, 100, 100), "WAITING"))
    cv2.putText(window, "Buffer Status:", (2 * col_w + 20, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)
    px = 2 * col_w + 120
    cv2.rectangle(window, (px, bar_y - 2), (px + 140, bar_y + 16), sc, -1)
    cv2.putText(window, st, (px + 8, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Confirmation and Speaking Overlay Notifications
    if confirm_flash:
        # Del action is colored in warning magenta, others in cyan/green
        flash_c = MAGENTA if confirm_flash == "DEL" else CYAN
        flash_lbl = "[ UNDO PREVIOUS ]" if confirm_flash == "DEL" else f"[ ADDED: {confirm_flash} ]"
        
        cv2.rectangle(window, (w_target - 150, 45), (w_target + 150, 85), flash_c, -1)
        # Text alignment centered
        t_sz = cv2.getTextSize(flash_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        tx = w_target - t_sz[0] // 2
        cv2.putText(window, flash_lbl, (tx, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    if speaking_flash:
        speak_text = f'SPEECH BROADCAST: "{speaking_flash}"'
        banner_y1 = h_target - 60
        banner_y2 = h_target
        
        # Translucent sound overlay
        overlay_banner = window.copy()
        cv2.rectangle(overlay_banner, (0, banner_y1), (combined_w, banner_y2), (22, 22, 22), -1)
        cv2.addWeighted(overlay_banner, 0.85, window, 0.15, 0, window)
        
        # Draw dynamic animated audio wave visualizer
        wave_center_y = h_target - 18
        wave_w = 260
        wave_start_x = (combined_w - wave_w) // 2
        phase = time.time() * 20
        prev_pt = None
        for dx in range(0, wave_w, 4):
            angle = (dx / wave_w) * 6 * math.pi + phase
            # Modulate amplitude using time to make it dynamic
            amp = 14 * math.sin(time.time() * 7) * math.sin(angle)
            x_val = wave_start_x + dx
            y_val = int(wave_center_y + amp)
            if prev_pt:
                cv2.line(window, prev_pt, (x_val, y_val), CYAN, 2)
            prev_pt = (x_val, y_val)
            
        # Draw text label above the wave
        t_sz = cv2.getTextSize(speak_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
        tx = (combined_w - t_sz[0]) // 2
        cv2.putText(window, speak_text, (tx, h_target - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, GREEN, 2, cv2.LINE_AA)

    return window


# ==================== MAIN LOOP ====================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try enhanced model first, fall back to original
    model_full_path = os.path.join(script_dir, MODEL_PATH_ENHANCED)
    encoder_full_path = os.path.join(script_dir, LABEL_ENCODER_PATH_ENHANCED)
    metadata_full_path = os.path.join(script_dir, METADATA_PATH)

    if not os.path.exists(model_full_path):
        print(f"Enhanced model not found, trying original model...")
        model_full_path = os.path.join(script_dir, MODEL_PATH_ORIGINAL)
        encoder_full_path = os.path.join(script_dir, LABEL_ENCODER_PATH_ORIGINAL)
        metadata_full_path = None

    # ---- Initialize pipeline components ----
    tracker = HandTracker(mode_config=MODE_PERFORMANCE)
    generator = SkeletonGenerator()
    engine = PredictionEngine(model_full_path, encoder_full_path, metadata_full_path)
    stability = StabilityManager()
    builder = WordBuilder(HOLD_TIME)
    fps_counter = FPSCounter()

    try:
        engine.load()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Setup webcam at 640×480 ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CRITICAL ERROR: Could not open webcam.", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ---- Runtime state ----
    prev_hand_detected = False
    frame_count = 0                   # For frame skipping logic
    last_raw_letter = None            # Cached prediction from skip frames
    last_confidence = 0.0
    last_top_3 = []                   # Cached Top 3 predictions
    last_canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255
    roi_box = None                    # Adaptive ROI from previous frame

    last_printed_state = (None, 0.0)

    # ---- TTS state ----
    speaking_flash_word = None        # Word currently being spoken (for HUD overlay)
    speaking_flash_time = 0           # Timestamp when speaking was triggered
    hand_absent_since = None          # time.time() when hand was last lost (for auto-speak)
    auto_speak_triggered = False      # Prevent repeated auto-speaks

    print("\n" + "=" * 55)
    print("Sign Language Predictor v5.0 — Production Pipeline")
    print("=" * 55)
    print("  [SPACE]  Manually add letter to buffer")
    print("  [C]      Clear word buffer")
    print("  [BACKSPACE] Delete last letter from buffer")
    print("  [ENTER]  Speak word buffer")
    print("  [P]      Performance mode (faster FPS)")
    print("  [A]      Accuracy mode (better detection)")
    print("  [ESC]    Quit")
    print(f"  Auto-Capture: Hold letter {HOLD_TIME}s to add automatically")
    print(f"  Current mode: {tracker.mode_config['name']}")
    print("=" * 55 + "\n")

    try:
        while True:
            fps_counter.tick()
            fps = fps_counter.get_fps()

            success, frame = cap.read()
            if not success:
                print("WARNING: Frame read failed.", file=sys.stderr)
                time.sleep(0.1)
                continue

            frame_h, frame_w, _ = frame.shape
            frame = cv2.flip(frame, 1)  # Mirror for natural perspective
            frame_count += 1

            # ---- Lighting normalization (conditional CLAHE) ----
            processed_frame, clahe_active = normalize_lighting(frame)

            # ---- Prepare RGB for MediaPipe ----
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # ---- MediaPipe hand detection ----
            hand_lms_obj, landmarks_visible = tracker.process_frame(rgb_frame)

            hand_detected = hand_lms_obj is not None
            raw_letter = last_raw_letter      # Default to cached (for skip frames)
            confidence = last_confidence
            canvas = last_canvas
            stable_pred = None
            stability_pct = 0.0
            tracking_status = "NO HAND"
            hand_motion = 0.0

            if hand_detected:
                # ---- Hand was just re-detected after absence ----
                if not prev_hand_detected:
                    tracker.reset()
                    stability.reset()

                current_lms = [(lm.x, lm.y, lm.z) for lm in hand_lms_obj.landmark]

                # ---- Motion measurement (before smoothing updates prev state) ----
                hand_motion = tracker.calculate_motion(current_lms)

                # ---- EMA landmark smoothing: 0.7 * prev + 0.3 * curr ----
                smoothed_lms = tracker.smooth_landmarks(current_lms)

                # ---- Cache for grace period recovery ----
                tracker.cached_landmarks = smoothed_lms

                # ---- Update adaptive ROI for potential future use ----
                roi_box = compute_roi(smoothed_lms, frame_w, frame_h)
                tracker.prev_crop_box = roi_box

                # ---- Generate normalized skeleton canvas ----
                canvas = generator.generate(smoothed_lms, frame_w, frame_h)
                last_canvas = canvas

                # ---- CNN prediction (with frame skipping) ----
                # Only run expensive CNN inference every FRAME_SKIP_INTERVAL frames.
                # On skipped frames, reuse the last prediction. Display updates every frame.
                if frame_count % FRAME_SKIP_INTERVAL == 0:
                    try:
                        pred_letter, confidence, top_3 = engine.predict(canvas)
                        raw_letter = pred_letter if confidence >= CONFIDENCE_THRESHOLD else "UNKNOWN"
                        last_raw_letter = raw_letter
                        last_confidence = confidence
                        last_top_3 = top_3
                    except Exception as e:
                        print(f"Inference error: {e}", file=sys.stderr)
                        raw_letter = "UNKNOWN"
                        last_raw_letter = raw_letter
                        last_top_3 = []

                # ---- Stability pipeline: motion gate → majority vote → gesture lock ----
                stable_pred, stability_pct, tracking_status = stability.update(
                    raw_letter, confidence, hand_motion
                )

            elif tracker.is_within_grace_period():
                # ---- Grace period: hand lost < 0.5s ago ----
                # Keep showing cached canvas and maintain stability state.
                # This prevents single-frame detection drops from resetting everything.
                tracking_status = "RECOVERY"
                canvas = last_canvas
                if stability.history:
                    counts = Counter(stability.history)
                    most_common, count = counts.most_common(1)[0]
                    stability_pct = float(count) / len(stability.history)
                    stable_pred = most_common
            else:
                # ---- Grace period expired — full reset ----
                tracker.reset()
                stability.reset()
                roi_box = None
                last_raw_letter = None
                last_confidence = 0.0
                last_top_3 = []
                last_canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255
                canvas = last_canvas

            prev_hand_detected = hand_detected

            # ---- Update word builder auto-capture ----
            builder.update(stable_pred, hand_detected or tracker.is_within_grace_period())

            # ---- Auto-speak logic: speak buffer when hand absent > AUTO_SPEAK_DELAY ----
            if hand_detected or tracker.is_within_grace_period():
                hand_absent_since = None
                auto_speak_triggered = False
            else:
                if hand_absent_since is None:
                    hand_absent_since = time.time()
                elif (not auto_speak_triggered
                      and builder.word_buffer.strip()
                      and (time.time() - hand_absent_since) >= AUTO_SPEAK_DELAY):
                    print(f"[TTS] Speaking: '{builder.word_buffer}'")
                    speak_text_async(builder.word_buffer)
                    speaking_flash_word = builder.word_buffer
                    speaking_flash_time = time.time()
                    builder.clear()
                    auto_speak_triggered = True

            # ---- Expire speaking flash after 2 seconds ----
            if speaking_flash_word and (time.time() - speaking_flash_time > 2.0):
                speaking_flash_word = None

            # ---- Tracking quality assessment for debug panel ----
            tracking_quality, tq_color = determine_tracking_quality(
                stability_pct, hand_motion, hand_detected
            )

            # ---- Console output (only on change) ----
            curr_state = (stable_pred, confidence)
            if stable_pred != last_printed_state[0] or abs(confidence - last_printed_state[1]) > 0.05:
                print(f"--- Prediction: {stable_pred or 'No hand'} | "
                      f"Conf: {confidence * 100:.1f}% | FPS: {int(fps)} | "
                      f"Mode: {tracker.mode_config['name']} ---")
                last_printed_state = curr_state

            # ---- Compose HUD ----
            combined_view = draw_ui(
                frame, canvas, raw_letter, confidence,
                stable_pred, stability_pct, builder.word_buffer, fps,
                len(stability.history), tracking_status, hand_detected,
                builder.capture_status, builder.get_elapsed_time(),
                builder.confirm_flash, tracker.mode_config["name"],
                landmarks_visible, tracking_quality, tq_color, clahe_active,
                last_top_3,
                speaking_flash_word,
            )

            cv2.imshow("Sign Language Predictor", combined_view)

            # ---- Key handling ----
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("ESC pressed. Quitting...")
                break
            elif key == ord(' '):  # Space manually adds letter
                builder.add_manual(stable_pred)
            elif key == 8:  # Backspace deletes last character
                builder.backspace()
            elif key == 13 or key == ord('s') or key == ord('S'):  # Enter or S speaks buffer
                if builder.word_buffer.strip():
                    print(f"[TTS] Speaking: '{builder.word_buffer}'")
                    speak_text_async(builder.word_buffer)
                    speaking_flash_word = builder.word_buffer
                    speaking_flash_time = time.time()
                    builder.clear()
            elif key == ord('c') or key == ord('C'):
                builder.clear()
            elif key == ord('p') or key == ord('P'):
                tracker.switch_mode(MODE_PERFORMANCE)
            elif key == ord('a') or key == ord('A'):
                tracker.switch_mode(MODE_ACCURACY)

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt. Closing...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("\nAll resources released. System closed.")


if __name__ == "__main__":
    main()
