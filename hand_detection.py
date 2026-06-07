import os
import sys
import warnings
import time
from utils import suppress_c_stderr

# Suppress internal MediaPipe C++ log messages (must be done before importing mediapipe)
os.environ['GLOG_minloglevel'] = '3'  # 3 is FATAL only, hiding warnings and errors

import cv2

# Initialize Mediapipe Hands with warnings suppressed
with suppress_c_stderr():
    import mediapipe as mp
    import numpy as np
    
    # Suppress protobuf deprecation warnings in terminal
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
    
    # Initialize Mediapipe Hands with high-accuracy configuration
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,        # False for video tracking (fast & smooth), True to run detection on every single frame (highest accuracy but slower)
        max_num_hands=2,                # Maximum number of hands to detect
        model_complexity=1,             # 1 for higher accuracy, 0 for lower latency
        min_detection_confidence=0.8,   # High threshold to filter false detections (default is 0.5)
        min_tracking_confidence=0.8     # High threshold to ensure high-accuracy tracking (default is 0.5)
    )
    
    # Run a dummy inference to trigger lazy-loaded TF Lite / XNNPACK initialization logs silently
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hands.process(dummy_frame)

# Class to smooth hand landmarks using Exponential Moving Average (EMA) to eliminate jitter
class HandSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        # List of previous hand landmarks stored as raw coordinate values: list of list of (x, y, z)
        self.prev_hands = []

    def smooth(self, current_hands):
        if not current_hands:
            self.prev_hands = []
            return current_hands

        new_prev_hands = []
        for curr_hand in current_hands:
            curr_wrist = curr_hand.landmark[0]
            best_match_idx = -1
            min_dist = 0.15  # Max wrist distance to consider it the same hand
            
            for idx, prev_hand in enumerate(self.prev_hands):
                prev_wrist_x, prev_wrist_y, _ = prev_hand[0]
                dist = ((curr_wrist.x - prev_wrist_x)**2 + (curr_wrist.y - prev_wrist_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx
            
            smoothed_coords = []
            if best_match_idx != -1:
                prev_hand = self.prev_hands[best_match_idx]
                for curr_lm, prev_lm in zip(curr_hand.landmark, prev_hand):
                    prev_x, prev_y, prev_z = prev_lm
                    # Apply EMA formula: new = alpha * current + (1 - alpha) * previous
                    curr_lm.x = self.alpha * curr_lm.x + (1 - self.alpha) * prev_x
                    curr_lm.y = self.alpha * curr_lm.y + (1 - self.alpha) * prev_y
                    curr_lm.z = self.alpha * curr_lm.z + (1 - self.alpha) * prev_z
                    smoothed_coords.append((curr_lm.x, curr_lm.y, curr_lm.z))
            else:
                for curr_lm in curr_hand.landmark:
                    smoothed_coords.append((curr_lm.x, curr_lm.y, curr_lm.z))
            
            new_prev_hands.append(smoothed_coords)
            
        self.prev_hands = new_prev_hands
        return current_hands

# Initialize drawing utils to draw skeleton/landmarks
mp_draw = mp.solutions.drawing_utils

# Custom neon drawing specs for a high-tech premium aesthetic (BGR format)
landmark_spec = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=-1, circle_radius=5)   # Glowing Neon Cyan joints
connection_spec = mp_draw.DrawingSpec(color=(180, 105, 255), thickness=3, circle_radius=2) # Neon Hot Pink bones

# Helper to draw a futuristic HUD card overlay on the frame
def draw_hud(frame, fps, hand_count):
    # Copy the frame to draw a semi-transparent box
    overlay = frame.copy()
    
    # Top-Left HUD dimensions
    x, y, w, h = 20, 20, 260, 105
    
    # Draw card body (Dark charcoal gray)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    
    # Draw neon border (Neon Cyan)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    # Alpha blend the overlay box back into the main frame (transparency)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # HUD Header Text
    cv2.putText(frame, "HUD: HAND MONITOR v1.0", (x + 12, y + 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Draw FPS details
    cv2.putText(frame, "FPS:", (x + 12, y + 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(fps)}", (x + 65, y + 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA) # Cyan
    
    # Draw Hand Count details
    cv2.putText(frame, "Hands:", (x + 12, y + 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Make color neon pink/magenta if any hand is detected, orange-red if no hands detected
    hand_color = (180, 105, 255) if hand_count > 0 else (100, 100, 255)
    cv2.putText(frame, f"{hand_count}", (x + 80, y + 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2, cv2.LINE_AA)

# Initialize smoother
smoother = HandSmoother(alpha=0.6)

# Initialize webcam and set resolution to 1280x720 for sharper hand details
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting hand detection. Press ESC to exit.")

# Variables to track and display a stable FPS (updated every 0.5 seconds)
frame_count = 0
fps_start_time = time.time()
display_fps = 0

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    result = hands.process(rgb)
    
    hand_count = 0
    # Smooth the hand landmark coordinates to remove jitter and increase precision
    if result.multi_hand_landmarks:
        hand_count = len(result.multi_hand_landmarks)
        result.multi_hand_landmarks = smoother.smooth(result.multi_hand_landmarks)

    # Draw the hand annotations on the image with custom neon style
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                landmark_spec,
                connection_spec
            )

    # Calculate real-time average FPS over a 0.5 second window to keep display stable
    frame_count += 1
    c_time = time.time()
    elapsed_time = c_time - fps_start_time
    if elapsed_time >= 0.5:
        display_fps = frame_count / elapsed_time
        frame_count = 0
        fps_start_time = c_time

    # Render HUD overlay card
    draw_hud(frame, display_fps, hand_count)

    # Display the stream
    cv2.imshow("Hand Detection", frame)

    # Break loop with 'ESC' key (ASCII 27)
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
