import os
import sys
import contextlib
import warnings

# ==================== SUPPRESS TF/MP C++ NOISE ====================
# Must be set before any TensorFlow or MediaPipe import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'


# Context manager to temporarily redirect low-level OS stderr (file descriptor 2) to devnull.
# This successfully suppresses C++ standard error prints from MediaPipe and TensorFlow.
@contextlib.contextmanager
def suppress_c_stderr():
    """Redirect file descriptor 2 (stderr) to devnull to silence low-level C++ logging."""
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError):
        yield
        return
    saved_stderr_fd = os.dup(stderr_fd)
    devnull = open(os.devnull, 'w')
    try:
        os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        devnull.close()


# Import heavy dependencies inside suppress context so C++ noise is silenced
with suppress_c_stderr():
    import numpy as np
    import cv2
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


class SkeletonGenerator:
    """
    Converts hand landmarks into a normalized 400×400 skeleton canvas that
    visually matches the training dataset format exactly:
      - White (255, 255, 255) background
      - Green (0, 255, 0) landmark dots with circle_radius=4
      - Green (0, 200, 0) connections with thickness=3
      - Hand centered and scaled to occupy ~75% of canvas

    The bounding box normalization ensures consistent CNN input regardless of
    where the hand appears in the webcam frame or how close/far it is.
    """

    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Enhanced drawing specs — thicker lines and larger dots for visual clarity
        # These match the training dataset's visual style exactly
        self.landmark_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=-1, circle_radius=4
        )
        self.connection_spec = self.mp_draw.DrawingSpec(
            color=(0, 200, 0), thickness=3, circle_radius=2
        )

    def generate(self, landmarks_list, frame_w, frame_h):
        """
        Transforms landmarks (list of (x, y, z) tuples) into a normalized skeleton canvas.

        Pipeline:
          1. Convert normalized coords → pixel coords using frame dimensions
          2. Compute tight bounding box around all 21 landmarks
          3. Add padding (20px) to prevent edge clipping
          4. Calculate uniform scale factor to fit hand into 300px (75% of 400)
          5. Translate hand center to canvas center (200, 200)
          6. Draw skeleton with dataset-matching green style

        Returns: numpy.ndarray (400, 400, 3) uint8 BGR image
        """
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Step 1: Bounding box in webcam pixel coordinates
        px_x = [pt[0] * frame_w for pt in landmarks_list]
        px_y = [pt[1] * frame_h for pt in landmarks_list]

        min_x, max_x = min(px_x), max(px_x)
        min_y, max_y = min(px_y), max(px_y)

        hand_w = max_x - min_x
        hand_h = max_y - min_y

        # Step 2: Padding for aspect ratio preservation
        padding = 20
        padded_w = hand_w + 2 * padding
        padded_h = hand_h + 2 * padding

        # Step 3: Uniform scale to occupy 75% of canvas (300 / max_padded_dimension)
        target_size = 300.0
        max_dim = max(padded_w, padded_h)
        scale = target_size / max_dim if max_dim > 0 else 1.0

        # Step 4: Hand center in pixel space
        center_x = min_x + hand_w / 2.0
        center_y = min_y + hand_h / 2.0

        # Step 5: Project each landmark onto canvas coordinates
        normalized_landmarks = landmark_pb2.NormalizedLandmarkList()
        for pt in landmarks_list:
            canvas_x = (pt[0] * frame_w - center_x) * scale + (self.width / 2.0)
            canvas_y = (pt[1] * frame_h - center_y) * scale + (self.height / 2.0)
            canvas_z = pt[2] * frame_w * scale

            new_lm = normalized_landmarks.landmark.add()
            new_lm.x = canvas_x / float(self.width)
            new_lm.y = canvas_y / float(self.height)
            new_lm.z = canvas_z / float(self.width)

        # Step 6: Draw skeleton
        self.mp_draw.draw_landmarks(
            canvas,
            normalized_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.landmark_spec,
            self.connection_spec,
        )
        return canvas


def extract_enhanced_features(hand_landmarks) -> np.ndarray:
    """
    Extracts 99 enhanced features from hand landmarks:
      1. Translates all landmarks relative to the wrist (landmark 0).
      2. Normalizes coordinates by dividing by wrist-to-middle-MCP (landmark 9) distance.
      3. Calculates:
         - Finger lengths (5 features)
         - Joint angles (14 angles)
         - Fingertip distances to wrist (5 features)
         - Pairwise fingertip distances (10 features)
         - Palm width (1 feature)
         - Palm height (1 feature)
         - Normalized coordinate vector (63 features)
    Returns:
      A flat 1D numpy array of size 99.
    """
    # Handle both MediaPipe landmark objects and raw arrays/lists of coords
    if hasattr(hand_landmarks, 'landmark'):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    else:
        coords = np.array(hand_landmarks, dtype=np.float32).reshape(21, 3)

    # 1. Translation: wrist is landmark 0
    wrist = coords[0]
    translated = coords - wrist  # shape: (21, 3)

    # 2. Scale normalization
    # Distance between wrist (0) and middle MCP (9)
    hand_scale = np.linalg.norm(translated[9])
    if hand_scale < 1e-6:
        hand_scale = 1.0
    normalized = translated / hand_scale

    # 3. Calculate Finger Lengths
    # Cumulative distances along the joints of each finger
    # Wrist (0), Thumb (1-4), Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    def finger_len(indices):
        return sum(np.linalg.norm(normalized[indices[i]] - normalized[indices[i+1]]) for i in range(len(indices)-1))

    lengths = [
        finger_len([0, 1, 2, 3, 4]),      # Thumb
        finger_len([0, 5, 6, 7, 8]),      # Index
        finger_len([0, 9, 10, 11, 12]),   # Middle
        finger_len([0, 13, 14, 15, 16]),  # Ring
        finger_len([0, 17, 18, 19, 20]),  # Pinky
    ]

    # 4. Joint angles
    # For three points A, B, C, angle at B is between BA and BC.
    # Angle is arccos of dot product of normalized BA and BC vectors.
    def joint_angle(idx_a, idx_b, idx_c):
        v1 = normalized[idx_a] - normalized[idx_b]
        v2 = normalized[idx_c] - normalized[idx_b]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    angles = [
        # Thumb
        joint_angle(1, 2, 3), joint_angle(2, 3, 4),
        # Index
        joint_angle(5, 6, 7), joint_angle(6, 7, 8),
        # Middle
        joint_angle(9, 10, 11), joint_angle(10, 11, 12),
        # Ring
        joint_angle(13, 14, 15), joint_angle(14, 15, 16),
        # Pinky
        joint_angle(17, 18, 19), joint_angle(18, 19, 20),
        # MCP angles (angle between wrist direction and first segment)
        joint_angle(0, 5, 6),
        joint_angle(0, 9, 10),
        joint_angle(0, 13, 14),
        joint_angle(0, 17, 18)
    ]

    # 5. Fingertip distances to wrist
    # Tips: 4, 8, 12, 16, 20
    tips = [4, 8, 12, 16, 20]
    tip_to_wrist = [np.linalg.norm(normalized[tip]) for tip in tips]

    # 6. Pairwise fingertip distances (10 combinations)
    tip_to_tip = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            tip_to_tip.append(np.linalg.norm(normalized[tips[i]] - normalized[tips[j]]))

    # 7. Palm width and height
    palm_width = np.linalg.norm(normalized[5] - normalized[17])
    palm_height = np.linalg.norm(normalized[0] - normalized[9])  # always 1.0 due to scale division

    # Concatenate all features into a single flat vector
    features = np.concatenate([
        normalized.flatten(),       # 63 features
        lengths,                    # 5 features
        angles,                     # 14 features
        tip_to_wrist,               # 5 features
        tip_to_tip,                 # 10 features
        [palm_width, palm_height]   # 2 features
    ])  # Total 99 features

    return features
