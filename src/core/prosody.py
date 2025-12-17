"""Prosody analysis module - extract audio features for emotion/intent detection."""

from pathlib import Path
from typing import Union, Optional
import numpy as np

from src.core.models import ProsodyFeatures


class ProsodyAnalyzer:
    """Extract prosody features from audio for emotion and intent analysis."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the prosody analyzer.
        
        Args:
            sample_rate: Expected sample rate of input audio
        """
        self.sample_rate = sample_rate
        self._librosa = None
    
    def _load_librosa(self):
        """Lazy load librosa."""
        if self._librosa is None:
            try:
                import librosa
                self._librosa = librosa
            except ImportError:
                raise ImportError(
                    "librosa not installed. Install with: pip install librosa"
                )
        return self._librosa
    
    def analyze_file(self, audio_path: Union[str, Path]) -> ProsodyFeatures:
        """
        Extract prosody features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            ProsodyFeatures with extracted features
        """
        librosa = self._load_librosa()
        
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        return self.analyze_array(y, sr)
    
    def analyze_array(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ProsodyFeatures:
        """
        Extract prosody features from audio array.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate (uses default if not provided)
            
        Returns:
            ProsodyFeatures with extracted features
        """
        librosa = self._load_librosa()
        sr = sample_rate or self.sample_rate
        
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Handle empty or very short audio
        if len(audio) < sr * 0.1:  # Less than 0.1 seconds
            return ProsodyFeatures()
        
        features = ProsodyFeatures()
        
        # Energy (RMS)
        try:
            rms = librosa.feature.rms(y=audio)
            features.energy = float(np.mean(rms))
        except Exception:
            features.energy = 0.0
        
        # Pitch (using YIN algorithm)
        try:
            pitches = librosa.yin(audio, fmin=50, fmax=300)
            # Filter out unreliable pitch estimates
            valid_pitches = pitches[(pitches > 50) & (pitches < 300)]
            if len(valid_pitches) > 0:
                features.pitch_mean = float(np.mean(valid_pitches))
                features.pitch_std = float(np.std(valid_pitches))
        except Exception:
            features.pitch_mean = 0.0
            features.pitch_std = 0.0
        
        # Tempo (speaking rate)
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.tempo = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
        except Exception:
            features.tempo = 0.0
        
        # Spectral centroid (brightness of sound)
        try:
            spectral = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.spectral_centroid = float(np.mean(spectral))
        except Exception:
            features.spectral_centroid = 0.0
        
        # Zero crossing rate (noisiness/roughness)
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.zero_crossing_rate = float(np.mean(zcr))
        except Exception:
            features.zero_crossing_rate = 0.0
        
        return features

