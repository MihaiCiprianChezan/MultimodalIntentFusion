"""Emotion detection module - analyze emotions from audio and video."""

from pathlib import Path
from typing import Union, Optional, Dict
import numpy as np

from src.core.models import EmotionResult, ProsodyFeatures


class AudioEmotionDetector:
    """Detect emotions from audio prosody features."""
    
    # Emotion thresholds based on prosody research
    EMOTION_PROFILES = {
        "angry": {"energy": (0.1, 1.0), "pitch_std": (30, 100), "tempo": (120, 200)},
        "happy": {"energy": (0.08, 0.2), "pitch_std": (20, 60), "tempo": (110, 160)},
        "sad": {"energy": (0.0, 0.06), "pitch_std": (0, 20), "tempo": (60, 100)},
        "fear": {"energy": (0.05, 0.15), "pitch_std": (25, 80), "tempo": (130, 200)},
        "neutral": {"energy": (0.03, 0.1), "pitch_std": (10, 30), "tempo": (90, 130)},
    }
    
    def detect(self, prosody: ProsodyFeatures) -> EmotionResult:
        """
        Detect emotion from prosody features using rule-based analysis.
        
        Args:
            prosody: Extracted prosody features
            
        Returns:
            EmotionResult with emotion probabilities
        """
        scores = {}
        
        for emotion, profile in self.EMOTION_PROFILES.items():
            score = 0.0
            
            # Check energy
            e_min, e_max = profile["energy"]
            if e_min <= prosody.energy <= e_max:
                score += 0.4
            elif prosody.energy < e_min:
                score += 0.2 * (prosody.energy / e_min) if e_min > 0 else 0
            
            # Check pitch variation
            p_min, p_max = profile["pitch_std"]
            if p_min <= prosody.pitch_std <= p_max:
                score += 0.3
            
            # Check tempo
            t_min, t_max = profile["tempo"]
            if t_min <= prosody.tempo <= t_max:
                score += 0.3
            
            scores[emotion] = max(0.0, min(1.0, score))
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        else:
            scores = {"neutral": 1.0}
        
        return EmotionResult.from_dict(scores)


class FacialEmotionDetector:
    """Detect emotions from facial expressions using DeepFace."""
    
    def __init__(self, detector_backend: str = "opencv"):
        """
        Initialize facial emotion detector.
        
        Args:
            detector_backend: Face detector backend ("opencv", "ssd", "mtcnn", "retinaface")
        """
        self.detector_backend = detector_backend
        self._deepface = None
    
    def _load_deepface(self):
        """Lazy load DeepFace."""
        if self._deepface is None:
            try:
                from deepface import DeepFace
                self._deepface = DeepFace
            except ImportError:
                raise ImportError(
                    "deepface not installed. Install with: pip install deepface"
                )
        return self._deepface
    
    def detect_from_image(self, image_path: Union[str, Path]) -> Optional[EmotionResult]:
        """
        Detect emotions from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            EmotionResult or None if no face detected
        """
        DeepFace = self._load_deepface()
        
        try:
            result = DeepFace.analyze(
                str(image_path),
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.detector_backend
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            return EmotionResult.from_dict(emotions)
            
        except Exception as e:
            print(f"⚠️ Face emotion detection failed: {e}")
            return None
    
    def detect_from_frame(self, frame: np.ndarray) -> Optional[EmotionResult]:
        """
        Detect emotions from video frame (numpy array).
        
        Args:
            frame: Video frame as numpy array (BGR format from OpenCV)
            
        Returns:
            EmotionResult or None if no face detected
        """
        DeepFace = self._load_deepface()
        
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.detector_backend
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            return EmotionResult.from_dict(emotions)
            
        except Exception:
            return None

