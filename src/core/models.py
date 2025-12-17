"""Data models for MultimodalIntentFusion."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum


class EmotionType(Enum):
    """Supported emotion types."""
    ANGRY = "angry"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


class GestureType(Enum):
    """Supported gesture types."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    POINTING = "pointing"
    WAVE = "wave"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    PEACE = "peace"
    OK = "ok"
    NONE = "none"


@dataclass
class ProsodyFeatures:
    """Audio prosody features extracted from speech."""
    energy: float = 0.0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    tempo: float = 0.0
    spectral_centroid: float = 0.0
    zero_crossing_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "energy": self.energy,
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "tempo": self.tempo,
            "spectral_centroid": self.spectral_centroid,
            "zero_crossing_rate": self.zero_crossing_rate,
        }
    
    def get_energy_description(self) -> str:
        """Get human-readable energy level."""
        if self.energy > 0.15:
            return "very energetic"
        elif self.energy > 0.08:
            return "energetic"
        elif self.energy > 0.03:
            return "moderate"
        else:
            return "calm"
    
    def get_tempo_description(self) -> str:
        """Get human-readable tempo description."""
        if self.tempo > 150:
            return "very fast"
        elif self.tempo > 120:
            return "fast"
        elif self.tempo > 90:
            return "moderate"
        else:
            return "slow"


@dataclass
class EmotionResult:
    """Emotion detection result."""
    probabilities: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = "neutral"
    confidence: float = 0.0
    
    @classmethod
    def from_dict(cls, probs: Dict[str, float]) -> "EmotionResult":
        if not probs:
            return cls(probabilities={"neutral": 1.0}, dominant_emotion="neutral", confidence=1.0)
        dominant = max(probs, key=probs.get)
        return cls(
            probabilities=probs,
            dominant_emotion=dominant,
            confidence=probs[dominant] / 100.0 if probs[dominant] > 1 else probs[dominant]
        )


@dataclass
class MultimodalSignal:
    """Container for all multimodal input signals."""
    text: str = ""
    prosody: Optional[ProsodyFeatures] = None
    audio_emotion: Optional[EmotionResult] = None
    facial_emotion: Optional[EmotionResult] = None
    gesture: Optional[str] = None
    context: str = ""
    
    def has_audio(self) -> bool:
        return bool(self.text) or self.prosody is not None
    
    def has_visual(self) -> bool:
        return self.facial_emotion is not None or self.gesture is not None


@dataclass  
class CompiledIntent:
    """The compiled intent output."""
    original_text: str
    compiled_prompt: str
    confidence: float
    modalities_used: List[str] = field(default_factory=list)
    emotion_context: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        return self.compiled_prompt

