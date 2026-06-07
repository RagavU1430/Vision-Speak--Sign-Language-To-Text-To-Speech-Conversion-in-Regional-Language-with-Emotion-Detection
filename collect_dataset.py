import os
import sys
import time
import warnings
from utils import suppress_c_stderr, SkeletonGenerator

# Suppress internal MediaPipe C++ log messages (must be done before importing mediapipe)
os.environ['GLOG_minloglevel'] = '3'

# Suppress warnings and load cv2 and mediapipe
with suppress_c_stderr():
    import cv2
    import mediapipe as mp
    import numpy as np
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Define target classes (A to Z)
GESTURES = {chr(i): chr(i) for i in range(ord('A'), ord('Z') + 1)}

TARGET_SAMPLES_PER_GESTURE = 2000
TARGET_TOTAL_SAMPLES = len(GESTURES) * TARGET_SAMPLES_PER_GESTURE  # 5200 total
DATASET_DIR = "archive"
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

def initialize_dataset_directory(base_dir):
    """
    Ensures that the directory hierarchy for A-Z labels exists under base_dir.
    Returns a dictionary mapping each gesture label to the number of existing files.
    """
    os.makedirs(base_dir, exist_ok=True)
    counts = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
    for char in counts.keys():
        char_dir = os.path.join(base_dir, char)
        os.makedirs(char_dir, exist_ok=True)
        try:
            valid_files = [f for f in os.listdir(char_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
            counts[char] = len(valid_files)
        except Exception:
            counts[char] = 0
    return counts

def save_sample(base_dir, label, landmarks_list, skeleton_gen):
    """
    Generates the skeleton canvas for the hand landmarks and saves it
    as a PNG image in the corresponding gesture directory.
    """
    try:
        # Generate the skeleton canvas (400x400)
        canvas = skeleton_gen.generate(landmarks_list, 1280, 720)  # Frame resolution
        
        # Save path
        label_dir = os.path.join(base_dir, label)
        timestamp = int(time.time() * 1000)
        img_name = f"sample_{timestamp}.png"
        img_path = os.path.join(label_dir, img_name)
        
        # Save image using OpenCV
        cv2.imwrite(img_path, canvas)
        return True
    except Exception as e:
        print(f"ERROR: Failed to save skeleton canvas image: {e}", file=sys.stderr)
        return False

def draw_ui(frame, current_label, sample_counts, status_message, message_expiry):
    """
    Draws a premium HUD overlay showing dataset collection details.
    """
    # Create semi-transparent overlay panel on the left for statistics
    overlay = frame.copy()
    
    # Left sidebar panel dimensions
    panel_w = 340
    panel_h = 420
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (25, 25, 25), -1)
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (255, 120, 0), 2)  # Neon Orange accent
    
    # Bottom keyboard shortcut panel
    cv2.rectangle(overlay, (15, frame.shape[0] - 65), (frame.shape[1] - 15, frame.shape[0] - 15), (20, 20, 20), -1)
    cv2.rectangle(overlay, (15, frame.shape[0] - 65), (frame.shape[1] - 15, frame.shape[0] - 15), (100, 100, 100), 1)

    # Apply alpha blending for glassmorphic transparent overlay
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw Left Sidebar HUD content
    cv2.putText(frame, "SIGN LANGUAGE DATA COLLECTOR", (30, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (30, 55), (15 + panel_w - 15, 55), (150, 150, 150), 1)

    # Total progress metrics
    total_samples = sum(sample_counts.values())
    progress_pct = min(100, int((total_samples / TARGET_TOTAL_SAMPLES) * 100))
    
    cv2.putText(frame, f"Total Samples: {total_samples} / {TARGET_TOTAL_SAMPLES}", (30, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if total_samples < TARGET_TOTAL_SAMPLES else (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw progress bar for total samples
    bar_x, bar_y, bar_w, bar_h = 30, 100, 280, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int((progress_pct / 100) * bar_w)
    bar_color = (0, 255, 0) if total_samples >= TARGET_TOTAL_SAMPLES else (255, 120, 0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    
    # Gesture Breakdown grid layout to support all 26 alphabets (A-Z)
    cv2.putText(frame, "GESTURE BREAKDOWN (Target: 200/ea)", (30, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    
    col_w = 75
    row_h = 34
    start_x = 30
    start_y = 175
    
    for idx, (key_shortcut, label) in enumerate(sorted(GESTURES.items())):
        col = idx % 4
        row = idx // 4
        
        x_pos = start_x + col * col_w
        y_pos = start_y + row * row_h
        
        count = sample_counts[label]
        if count >= TARGET_SAMPLES_PER_GESTURE:
            text_color = (100, 255, 100)
        elif count > 0:
            text_color = (0, 255, 255)
        else:
            text_color = (180, 180, 180)
            
        display_text = f"{label}:{count}"
        cv2.putText(frame, display_text, (x_pos, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # Draw Bottom Keyboard Shortcuts reference
    shortcuts_str = "Press keys [A-Z] to capture corresponding gesture | [ESC] -> Exit"
    cv2.putText(frame, shortcuts_str, (30, frame.shape[0] - 32), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Active selected gesture indicator (top-right of screen)
    if current_label:
        cv2.rectangle(frame, (frame.shape[1] - 220, 20), (frame.shape[1] - 20, 70), (40, 40, 40), -1)
        cv2.rectangle(frame, (frame.shape[1] - 220, 20), (frame.shape[1] - 20, 70), (0, 255, 255), 2)
        cv2.putText(frame, f"ACTIVE: {current_label.upper()}", (frame.shape[1] - 205, 52), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # Display dynamic confirmation / warning status messages (centered at the bottom)
    if time.time() < message_expiry:
        if "WARNING" in status_message or "No hand" in status_message:
            text_color = (0, 0, 255)  # Red
            bg_color = (10, 10, 50)
        else:
            text_color = (0, 255, 0)  # Green
            bg_color = (10, 50, 10)
            
        msg_size = cv2.getTextSize(status_message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        msg_x = (frame.shape[1] - msg_size[0]) // 2
        msg_y = frame.shape[0] - 100
        
        cv2.rectangle(frame, (msg_x - 15, msg_y - 25), (msg_x + msg_size[0] + 15, msg_y + 10), bg_color, -1)
        cv2.rectangle(frame, (msg_x - 15, msg_y - 25), (msg_x + msg_size[0] + 15, msg_y + 10), text_color, 1)
        cv2.putText(frame, status_message, (msg_x, msg_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

def main():
    # 1. Initialize Dataset directory structure and counts
    sample_counts = initialize_dataset_directory(DATASET_DIR)
    
    # UI temporary alert message states
    status_message = ""
    message_expiry = 0.0
    
    # 2. Setup OpenCV Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CRITICAL ERROR: Could not access the webcam. Please ensure it is connected and not in use.", file=sys.stderr)
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 3. Setup MediaPipe Hands Solution
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    landmark_spec = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=-1, circle_radius=5)   # Neon Cyan joints
    connection_spec = mp_draw.DrawingSpec(color=(180, 105, 255), thickness=3, circle_radius=2) # Neon Pink bones
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )
    
    skeleton_gen = SkeletonGenerator(width=400, height=400)
    
    # 4. Helper tracking states
    current_label = None
    last_processed_landmarks = None
    hand_detected_in_frame = False
    
    print("\n" + "="*50)
    print("Sign Language A-Z Gesture Collection System Active.")
    print("Use keyboard keys [A-Z] to save gestures:")
    print("  [ESC] -> Quit & Save cleanly")
    print("="*50 + "\n")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("WARNING: Empty webcam frame received. Attempting to continue...", file=sys.stderr)
                time.sleep(0.1)
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            hand_detected_in_frame = False
            last_processed_landmarks = None
            
            if results.multi_hand_landmarks:
                hand_detected_in_frame = True
                first_hand = results.multi_hand_landmarks[0]
                
                # Draw neon skeleton overlay on frame preview
                mp_draw.draw_landmarks(
                    frame,
                    first_hand,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_spec,
                    connection_spec
                )
                
                # Extract landmarks list for skeleton canvas generation
                last_processed_landmarks = [(lm.x, lm.y, lm.z) for lm in first_hand.landmark]
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC Key
                print("Exit signal received. Terminating program cleanly...")
                break
                
            pressed_char = chr(key).upper() if key < 256 else ""
            if pressed_char in GESTURES:
                label_to_save = GESTURES[pressed_char]
                current_label = label_to_save
                
                if hand_detected_in_frame and last_processed_landmarks is not None:
                    # Save skeleton canvas directly to images folder
                    save_success = save_sample(DATASET_DIR, label_to_save, last_processed_landmarks, skeleton_gen)
                    if save_success:
                        sample_counts[label_to_save] += 1
                        total_cnt = sum(sample_counts.values())
                        status_message = f"Saved '{label_to_save}' sample #{sample_counts[label_to_save]} (Total: {total_cnt})"
                    else:
                        status_message = "CRITICAL ERROR: Failed to save image!"
                else:
                    status_message = "WARNING: No hand detected! Position hand in frame."
                
                message_expiry = time.time() + 1.5
                
            draw_ui(frame, current_label, sample_counts, status_message, message_expiry)
            cv2.imshow("Sign Language Gesture Collector", frame)
            
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Exiting gracefully...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\nAll resources released successfully.")
        print(f"Final collected data location: {os.path.abspath(DATASET_DIR)}")
        print(f"Total dataset size: {sum(sample_counts.values())} samples.")

if __name__ == "__main__":
    main()
