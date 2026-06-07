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
