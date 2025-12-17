"""Configuration settings for MultimodalIntentFusion."""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class WhisperConfig:
    """Whisper speech recognition configuration."""
    model_size: str = "base.en"
    # Options: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large
    language: str = "en"
    sample_rate: int = 16000


@dataclass
class LLMConfig:
    """LLM configuration for intent compilation."""
    provider: str = "ollama"  # ollama, openai, anthropic
    model: str = "llama3.1:8b"
    # Good options: llama3.1:8b, mistral:7b, qwen3:8b, deepseek-r1:8b
    temperature: float = 0.7
    max_tokens: int = 500


@dataclass
class ProsodyConfig:
    """Prosody analysis configuration."""
    sample_rate: int = 16000
    pitch_fmin: float = 50.0
    pitch_fmax: float = 300.0


@dataclass
class EmotionConfig:
    """Emotion detection configuration."""
    # Audio emotion detection
    audio_enabled: bool = True
    
    # Facial emotion detection
    facial_enabled: bool = False
    face_detector_backend: str = "opencv"  # opencv, ssd, mtcnn, retinaface


@dataclass
class VisionConfig:
    """Vision/gesture configuration."""
    enabled: bool = False
    camera_id: int = 0
    max_hands: int = 2
    detection_confidence: float = 0.5


@dataclass
class AppConfig:
    """Main application configuration."""
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    
    # Recording settings
    default_record_duration: float = 5.0
    
    # Output settings  
    verbose: bool = True
    
    @classmethod
    def load_from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        import os
        
        config = cls()
        
        # Override from environment
        if whisper_model := os.getenv("MIF_WHISPER_MODEL"):
            config.whisper.model_size = whisper_model
        
        if llm_model := os.getenv("MIF_LLM_MODEL"):
            config.llm.model = llm_model
        
        if os.getenv("MIF_ENABLE_VISION", "").lower() == "true":
            config.vision.enabled = True
            config.emotion.facial_enabled = True
        
        return config


# Default configuration instance
default_config = AppConfig()

