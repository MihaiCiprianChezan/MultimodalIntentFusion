"""Tests for data models."""

import pytest
from src.core.models import (
    ProsodyFeatures,
    EmotionResult,
    MultimodalSignal,
    CompiledIntent,
    EmotionType,
    GestureType,
)


class TestProsodyFeatures:
    """Tests for ProsodyFeatures dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        features = ProsodyFeatures()
        assert features.energy == 0.0
        assert features.pitch_mean == 0.0
        assert features.tempo == 0.0
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        features = ProsodyFeatures(energy=0.5, pitch_mean=150.0, tempo=120.0)
        result = features.to_dict()
        assert result["energy"] == 0.5
        assert result["pitch_mean"] == 150.0
        assert result["tempo"] == 120.0
    
    def test_energy_description(self):
        """Test energy level descriptions."""
        assert ProsodyFeatures(energy=0.02).get_energy_description() == "calm"
        assert ProsodyFeatures(energy=0.05).get_energy_description() == "moderate"
        assert ProsodyFeatures(energy=0.10).get_energy_description() == "energetic"
        assert ProsodyFeatures(energy=0.20).get_energy_description() == "very energetic"
    
    def test_tempo_description(self):
        """Test tempo descriptions."""
        assert ProsodyFeatures(tempo=80).get_tempo_description() == "slow"
        assert ProsodyFeatures(tempo=100).get_tempo_description() == "moderate"
        assert ProsodyFeatures(tempo=130).get_tempo_description() == "fast"
        assert ProsodyFeatures(tempo=160).get_tempo_description() == "very fast"


class TestEmotionResult:
    """Tests for EmotionResult dataclass."""
    
    def test_from_dict(self):
        """Test creation from probability dictionary."""
        probs = {"happy": 0.6, "neutral": 0.3, "sad": 0.1}
        result = EmotionResult.from_dict(probs)
        assert result.dominant_emotion == "happy"
        assert result.confidence == 0.6
    
    def test_from_dict_percentage(self):
        """Test creation from percentage values (DeepFace style)."""
        probs = {"happy": 60.0, "neutral": 30.0, "sad": 10.0}
        result = EmotionResult.from_dict(probs)
        assert result.dominant_emotion == "happy"
        assert result.confidence == 0.6
    
    def test_empty_dict(self):
        """Test handling of empty input."""
        result = EmotionResult.from_dict({})
        assert result.dominant_emotion == "neutral"
        assert result.confidence == 1.0


class TestMultimodalSignal:
    """Tests for MultimodalSignal dataclass."""
    
    def test_has_audio(self):
        """Test audio detection."""
        signal = MultimodalSignal(text="hello")
        assert signal.has_audio() is True
        
        signal_empty = MultimodalSignal()
        assert signal_empty.has_audio() is False
        
        signal_prosody = MultimodalSignal(prosody=ProsodyFeatures())
        assert signal_prosody.has_audio() is True
    
    def test_has_visual(self):
        """Test visual signal detection."""
        signal = MultimodalSignal()
        assert signal.has_visual() is False
        
        signal_face = MultimodalSignal(
            facial_emotion=EmotionResult(dominant_emotion="happy")
        )
        assert signal_face.has_visual() is True
        
        signal_gesture = MultimodalSignal(gesture="thumbs_up")
        assert signal_gesture.has_visual() is True


class TestCompiledIntent:
    """Tests for CompiledIntent dataclass."""
    
    def test_str_method(self):
        """Test string representation."""
        intent = CompiledIntent(
            original_text="um can you help me",
            compiled_prompt="Please help me with...",
            confidence=0.9
        )
        assert str(intent) == "Please help me with..."
    
    def test_modalities_tracking(self):
        """Test modalities list."""
        intent = CompiledIntent(
            original_text="test",
            compiled_prompt="test compiled",
            confidence=1.0,
            modalities_used=["speech", "prosody", "emotion"]
        )
        assert "speech" in intent.modalities_used
        assert len(intent.modalities_used) == 3

