"""Speech-to-Text module using Whisper."""

import os
from pathlib import Path
from typing import Optional, Union
import numpy as np


class SpeechRecognizer:
    """Speech-to-text using faster-whisper (local, free, fast)."""

    def __init__(self, model_size: str = "base.en"):
        """
        Initialize the speech recognizer.

        Args:
            model_size: Whisper model size. Options:
                - "tiny.en" / "tiny": ~75MB, fastest
                - "base.en" / "base": ~142MB, good balance (default)
                - "small.en" / "small": ~466MB, better accuracy
                - "medium.en" / "medium": ~1.5GB, high accuracy
                - "large-v3": ~2.9GB, best accuracy
        """
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                print(f"🎤 Loading Whisper model ({self.model_size})...")
                # Use CPU with int8 quantization for efficiency
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                print("✅ Whisper loaded!")
            except ImportError:
                raise ImportError(
                    "faster-whisper not installed. Install with: pip install faster-whisper"
                )
        return self._model

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            Transcribed text
        """
        model = self._load_model()
        audio_path = str(audio_path)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = model.transcribe(audio_path, beam_size=5)

        # Combine all segments into one text
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        return " ".join(text_parts).strip()

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio from numpy array.

        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        import tempfile
        import scipy.io.wavfile as wav

        # Ensure audio is in correct format (int16)
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            audio_int = (audio_array * 32767).astype(np.int16)
        else:
            audio_int = audio_array.astype(np.int16)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            wav.write(temp_path, sample_rate, audio_int)

        try:
            return self.transcribe(temp_path)
        finally:
            os.unlink(temp_path)


class AudioRecorder:
    """Record audio from microphone."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
    
    def record(self, duration: float = 5.0) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Audio samples as numpy array
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice not installed. Install with: pip install sounddevice"
            )
        
        print(f"🎙️ Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()
        print("✅ Recording complete!")
        
        return audio.flatten()
    
    def save(self, audio: np.ndarray, path: Union[str, Path]) -> None:
        """Save audio array to file."""
        import scipy.io.wavfile as wav
        
        # Convert to int16 for WAV
        audio_int = (audio * 32767).astype(np.int16)
        wav.write(str(path), self.sample_rate, audio_int)

