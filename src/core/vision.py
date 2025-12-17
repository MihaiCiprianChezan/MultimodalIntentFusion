"""Vision module - gesture recognition and visual analysis."""

from typing import Optional, List, Tuple
import numpy as np

from src.core.models import GestureType


class GestureRecognizer:
    """
    Recognize hand gestures using OpenCV-based skin detection.

    This is a simple fallback that doesn't require MediaPipe (which has
    dependency conflicts with TensorFlow on Windows).
    """

    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5):
        """
        Initialize gesture recognizer.

        Args:
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self._cv2 = None

    def _load_opencv(self):
        """Lazy load OpenCV."""
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                raise ImportError(
                    "opencv-python not installed. Install with: pip install opencv-python"
                )
        return self._cv2

    def detect(self, frame: np.ndarray) -> List[str]:
        """
        Detect gestures in video frame using skin color detection.

        Args:
            frame: Video frame as numpy array (BGR format)

        Returns:
            List of detected gesture names
        """
        cv2 = self._load_opencv()

        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin color range in HSV (works for various skin tones)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask for skin regions
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours (potential hands)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gestures = []

        # Process the largest contours (likely hands)
        if contours:
            # Sort by area, largest first
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in sorted_contours[:self.max_hands]:
                area = cv2.contourArea(contour)

                # Filter by minimum area (hand should be reasonably large)
                if area < 5000:
                    continue

                # Analyze the contour shape
                gesture = self._classify_contour(contour, cv2)
                if gesture:
                    gestures.append(gesture)

        return gestures

    def _classify_contour(self, contour, cv2) -> Optional[str]:
        """
        Classify gesture based on contour analysis.

        Uses convexity defects to count fingers.
        """
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)

        if len(hull) < 3:
            return None

        try:
            # Find convexity defects
            defects = cv2.convexityDefects(contour, hull)

            if defects is None:
                return GestureType.FIST.value

            # Count significant defects (spaces between fingers)
            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]

                # d is the distance from the farthest point to the hull
                # Large distances indicate spaces between fingers
                if d > 10000:  # Threshold for significant defect
                    finger_count += 1

            # Classify based on finger count
            if finger_count == 0:
                return GestureType.FIST.value
            elif finger_count == 1:
                return GestureType.POINTING.value
            elif finger_count == 2:
                return GestureType.PEACE.value
            elif finger_count >= 4:
                return GestureType.OPEN_PALM.value
            else:
                return None

        except Exception:
            return None

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

