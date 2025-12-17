"""Vision module - gesture recognition and visual analysis."""

from typing import Optional, List, Tuple
import numpy as np
import os
import urllib.request

from src.core.models import GestureType


class GestureRecognizer:
    """Recognize hand gestures using MediaPipe Tasks API."""

    # Model URL for hand landmarker
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5):
        """
        Initialize gesture recognizer.

        Args:
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self._landmarker = None
        self._last_result = None

    def _download_model(self):
        """Download the hand landmarker model if not present."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"📥 Downloading hand landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"✅ Model downloaded to {self.MODEL_PATH}")

    def _load_mediapipe(self):
        """Lazy load MediaPipe Hand Landmarker."""
        if self._landmarker is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                # Download model if needed
                self._download_model()

                # Create hand landmarker
                base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=self.max_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_detection_confidence,
                    running_mode=vision.RunningMode.IMAGE
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
                self._mp = mp

            except ImportError as e:
                raise ImportError(
                    f"mediapipe not installed correctly. Install with: pip install mediapipe\nError: {e}"
                )
        return self._landmarker

    def detect(self, frame: np.ndarray) -> List[str]:
        """
        Detect gestures in video frame.

        Args:
            frame: Video frame as numpy array (BGR format)

        Returns:
            List of detected gesture names
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python not installed. Install with: pip install opencv-python"
            )

        landmarker = self._load_mediapipe()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        result = landmarker.detect(mp_image)
        self._last_result = result

        gestures = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                gesture = self._classify_gesture(hand_landmarks)
                if gesture:
                    gestures.append(gesture)

        return gestures

    def _classify_gesture(self, hand_landmarks) -> Optional[str]:
        """
        Classify gesture from hand landmarks.

        Args:
            hand_landmarks: List of NormalizedLandmark from MediaPipe Tasks API

        Returns:
            Gesture name or None
        """
        # hand_landmarks is a list of NormalizedLandmark objects
        landmarks = hand_landmarks

        # Finger tip and base indices
        THUMB_TIP, THUMB_IP = 4, 3
        INDEX_TIP, INDEX_PIP = 8, 6
        MIDDLE_TIP, MIDDLE_PIP = 12, 10
        RING_TIP, RING_PIP = 16, 14
        PINKY_TIP, PINKY_PIP = 20, 18

        # Check if fingers are extended
        def is_finger_extended(tip_idx: int, pip_idx: int) -> bool:
            return landmarks[tip_idx].y < landmarks[pip_idx].y

        thumb_extended = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x  # Simplified
        index_extended = is_finger_extended(INDEX_TIP, INDEX_PIP)
        middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_PIP)
        ring_extended = is_finger_extended(RING_TIP, RING_PIP)
        pinky_extended = is_finger_extended(PINKY_TIP, PINKY_PIP)

        fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        extended_count = sum(fingers)

        # Classify gestures
        if extended_count == 0:
            return GestureType.FIST.value

        if extended_count == 5:
            return GestureType.OPEN_PALM.value

        if thumb_extended and not any(fingers[1:]):
            return GestureType.THUMBS_UP.value

        if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            return GestureType.POINTING.value

        if index_extended and middle_extended and not any([ring_extended, pinky_extended]):
            return GestureType.PEACE.value

        if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            # Check if forming OK sign (thumb and index touching)
            thumb_tip = landmarks[THUMB_TIP]
            index_tip = landmarks[INDEX_TIP]
            distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            if distance < 0.05:
                return GestureType.OK.value
        
        return GestureType.NONE.value
    
    def close(self):
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()
            self._hands = None


class WebcamCapture:
    """Capture frames from webcam."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self._cap = None
    
    def _open(self):
        """Open webcam."""
        if self._cap is None:
            try:
                import cv2
                self._cap = cv2.VideoCapture(self.camera_id)
            except ImportError:
                raise ImportError(
                    "opencv-python not installed. Install with: pip install opencv-python"
                )
        return self._cap
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame from webcam."""
        cap = self._open()
        ret, frame = cap.read()
        return frame if ret else None
    
    def close(self):
        """Release webcam."""
        if self._cap:
            self._cap.release()
            self._cap = None

