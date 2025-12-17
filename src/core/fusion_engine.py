"""Main fusion engine - combines all modalities into compiled intent."""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from src.core.models import MultimodalSignal, CompiledIntent, ProsodyFeatures, EmotionResult
from src.core.speech import SpeechRecognizer, AudioRecorder
from src.core.prosody import ProsodyAnalyzer
from src.core.emotion import AudioEmotionDetector, FacialEmotionDetector
from src.core.vision import GestureRecognizer, WebcamCapture
from src.core.llm import IntentCompiler, OllamaLLM, LlamaCppLLM, get_default_llm


class MultimodalIntentFusion:
    """
    Main engine for multimodal intent fusion.

    Combines speech, prosody, emotion, and visual signals
    to create clear, machine-ready prompts.
    """

    def __init__(
        self,
        whisper_model: str = "base.en",
        llm_model: str = "llama3.1:8b",
        llamacpp_url: str = "http://localhost:8082",
        enable_vision: bool = False,
        lazy_load: bool = True
    ):
        """
        Initialize the fusion engine.

        Args:
            whisper_model: Whisper model size for speech recognition
            llm_model: Ollama model for intent compilation (if using Ollama)
            llamacpp_url: URL for llama.cpp server (preferred over Ollama)
            enable_vision: Enable webcam/gesture features
            lazy_load: Lazy load models (recommended)
        """
        self.whisper_model = whisper_model
        self.llm_model = llm_model
        self.llamacpp_url = llamacpp_url
        self.enable_vision = enable_vision

        # Components (lazy loaded)
        self._speech = None
        self._prosody = None
        self._audio_emotion = None
        self._facial_emotion = None
        self._gesture = None
        self._webcam = None
        self._compiler = None
        self._recorder = None
        
        if not lazy_load:
            self._initialize_all()
    
    def _initialize_all(self):
        """Initialize all components."""
        self.speech
        self.prosody
        self.audio_emotion
        self.compiler
        if self.enable_vision:
            self.facial_emotion
            self.gesture
    
    @property
    def speech(self) -> SpeechRecognizer:
        if self._speech is None:
            self._speech = SpeechRecognizer(self.whisper_model)
        return self._speech
    
    @property
    def prosody(self) -> ProsodyAnalyzer:
        if self._prosody is None:
            self._prosody = ProsodyAnalyzer()
        return self._prosody
    
    @property
    def audio_emotion(self) -> AudioEmotionDetector:
        if self._audio_emotion is None:
            self._audio_emotion = AudioEmotionDetector()
        return self._audio_emotion
    
    @property
    def facial_emotion(self) -> FacialEmotionDetector:
        if self._facial_emotion is None:
            self._facial_emotion = FacialEmotionDetector()
        return self._facial_emotion
    
    @property
    def gesture(self) -> GestureRecognizer:
        if self._gesture is None:
            self._gesture = GestureRecognizer()
        return self._gesture
    
    @property
    def webcam(self) -> WebcamCapture:
        if self._webcam is None:
            self._webcam = WebcamCapture()
        return self._webcam
    
    @property
    def compiler(self) -> IntentCompiler:
        if self._compiler is None:
            # Auto-detect best available LLM (prefers llama.cpp)
            llm = get_default_llm(
                llamacpp_url=self.llamacpp_url,
                ollama_model=self.llm_model
            )
            self._compiler = IntentCompiler(llm)
        return self._compiler
    
    @property
    def recorder(self) -> AudioRecorder:
        if self._recorder is None:
            self._recorder = AudioRecorder()
        return self._recorder
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        video_frame: Optional[np.ndarray] = None,
        context: str = ""
    ) -> CompiledIntent:
        """
        Process audio file through the full pipeline.
        
        Args:
            audio_path: Path to audio file
            video_frame: Optional video frame for facial analysis
            context: Additional context
            
        Returns:
            CompiledIntent with the processed result
        """
        modalities = []
        
        # 1. Speech-to-Text
        print("📝 Transcribing speech...")
        text = self.speech.transcribe(audio_path)
        modalities.append("speech")
        print(f"   Text: {text}")
        
        # 2. Prosody Analysis
        print("🎵 Analyzing prosody...")
        prosody = self.prosody.analyze_file(audio_path)
        modalities.append("prosody")
        print(f"   Energy: {prosody.get_energy_description()}, Tempo: {prosody.get_tempo_description()}")
        
        # 3. Audio Emotion
        print("🎭 Detecting audio emotion...")
        audio_emo = self.audio_emotion.detect(prosody)
        modalities.append("audio_emotion")
        print(f"   Emotion: {audio_emo.dominant_emotion} ({audio_emo.confidence:.0%})")
        
        # 4. Visual Analysis (optional)
        facial_emo = None
        detected_gesture = None
        
        if video_frame is not None and self.enable_vision:
            print("👁️ Analyzing visual signals...")
            facial_emo = self.facial_emotion.detect_from_frame(video_frame)
            if facial_emo:
                modalities.append("facial_emotion")
                print(f"   Facial: {facial_emo.dominant_emotion}")
            
            gestures = self.gesture.detect(video_frame)
            if gestures:
                detected_gesture = gestures[0]
                modalities.append("gesture")
                print(f"   Gesture: {detected_gesture}")
        
        # 5. Compile Intent
        print("✨ Compiling intent...")
        speaking_style = f"{prosody.get_energy_description()}, {prosody.get_tempo_description()} pace"
        
        compiled_text = self.compiler.compile(
            text=text,
            context=context,
            emotion=audio_emo.dominant_emotion,
            speaking_style=speaking_style,
            gesture=detected_gesture
        )
        
        return CompiledIntent(
            original_text=text,
            compiled_prompt=compiled_text,
            confidence=audio_emo.confidence,
            modalities_used=modalities,
            emotion_context=audio_emo.dominant_emotion,
            metadata={
                "prosody": prosody.to_dict(),
                "audio_emotion": audio_emo.probabilities,
                "facial_emotion": facial_emo.probabilities if facial_emo else None,
                "gesture": detected_gesture
            }
        )

    def process_text(self, text: str, context: str = "") -> CompiledIntent:
        """
        Process text-only input (MVP mode).

        Args:
            text: Raw text input
            context: Additional context

        Returns:
            CompiledIntent with the processed result
        """
        print("✨ Compiling intent from text...")

        compiled_text = self.compiler.compile(text=text, context=context)

        return CompiledIntent(
            original_text=text,
            compiled_prompt=compiled_text,
            confidence=1.0,
            modalities_used=["text"],
            metadata={}
        )

    def record_and_process(
        self,
        duration: float = 5.0,
        capture_video: bool = False,
        context: str = ""
    ) -> CompiledIntent:
        """
        Record from microphone and process.

        Args:
            duration: Recording duration in seconds
            capture_video: Also capture webcam frame
            context: Additional context

        Returns:
            CompiledIntent with the processed result
        """
        import tempfile
        import scipy.io.wavfile as wav

        # Record audio
        audio = self.recorder.record(duration)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_int = (audio * 32767).astype(np.int16)
            wav.write(temp_path, self.recorder.sample_rate, audio_int)

        # Capture video frame if enabled
        video_frame = None
        if capture_video and self.enable_vision:
            video_frame = self.webcam.capture_frame()

        try:
            return self.process_audio(temp_path, video_frame, context)
        finally:
            import os
            os.unlink(temp_path)

    def close(self):
        """Release resources."""
        if self._webcam:
            self._webcam.close()
        if self._gesture:
            self._gesture.close()

