#!/usr/bin/env python
"""
Interactive Multimodal Intent Fusion Demo

Captures real input from:
- 🎤 Microphone (speech → text + prosody + emotion)
- 📷 Webcam (facial emotion + gestures)

Fuses everything into a single enriched prompt that helps AI understand the user better.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile
import threading
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

console = Console()


class InteractiveMultimodalDemo:
    """Interactive demo capturing all modalities."""
    
    def __init__(self, llamacpp_url: str = "http://localhost:8082"):
        self.llamacpp_url = llamacpp_url
        
        # Lazy load components
        self._speech = None
        self._prosody = None
        self._audio_emotion = None
        self._facial_emotion = None
        self._gesture = None
        self._compiler = None
        self._webcam = None
        
        # Captured data
        self.video_frames = []
        self.capture_video = True
    
    def _init_components(self):
        """Initialize all components."""
        console.print("[dim]Loading components...[/dim]")
        
        from src.core.speech import SpeechRecognizer, AudioRecorder
        from src.core.prosody import ProsodyAnalyzer
        from src.core.emotion import AudioEmotionDetector, FacialEmotionDetector
        from src.core.vision import GestureRecognizer, WebcamCapture
        from src.core.llm import IntentCompiler, LlamaCppLLM
        
        self._recorder = AudioRecorder()
        self._speech = SpeechRecognizer()
        self._prosody = ProsodyAnalyzer()
        self._audio_emotion = AudioEmotionDetector()
        self._facial_emotion = FacialEmotionDetector()
        self._gesture = GestureRecognizer()
        self._webcam = WebcamCapture()
        self._compiler = IntentCompiler(LlamaCppLLM(base_url=self.llamacpp_url))
        
        console.print("[green]✅ All components loaded![/green]\n")
    
    def _capture_video_thread(self, duration: float):
        """Capture video frames in background thread."""
        self.video_frames = []
        start_time = time.time()
        
        while time.time() - start_time < duration and self.capture_video:
            frame = self._webcam.capture_frame()
            if frame is not None:
                self.video_frames.append(frame)
            time.sleep(0.1)  # ~10 FPS
    
    def record_and_analyze(self, duration: float = 5.0):
        """Record audio + video and analyze all modalities."""
        import scipy.io.wavfile as wav
        
        console.print(f"\n[bold yellow]🎤 Recording for {duration} seconds...[/bold yellow]")
        console.print("[dim]Speak naturally and use gestures![/dim]\n")
        
        # Start video capture in background
        self.capture_video = True
        video_thread = threading.Thread(target=self._capture_video_thread, args=(duration,))
        video_thread.start()
        
        # Record audio
        audio = self._recorder.record(duration)
        
        # Stop video capture
        self.capture_video = False
        video_thread.join()
        
        console.print("[green]✅ Recording complete![/green]\n")
        
        # Save audio to temp file for speech recognition
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_int = (audio * 32767).astype(np.int16)
            wav.write(temp_path, self._recorder.sample_rate, audio_int)
        
        results = {
            "text": "",
            "prosody": None,
            "audio_emotion": None,
            "facial_emotion": None,
            "gestures": [],
        }
        
        # 1. Speech-to-Text
        console.print("[cyan]📝 Transcribing speech...[/cyan]")
        try:
            results["text"] = self._speech.transcribe(temp_path)
            console.print(f"   Text: \"{results['text']}\"")
        except Exception as e:
            console.print(f"[yellow]   ⚠️ Speech recognition failed: {e}[/yellow]")
            results["text"] = ""
        
        # 2. Prosody Analysis
        console.print("[cyan]🎵 Analyzing prosody...[/cyan]")
        try:
            results["prosody"] = self._prosody.analyze_array(audio, self._recorder.sample_rate)
            console.print(f"   Energy: {results['prosody'].energy:.3f}, Pitch: {results['prosody'].pitch_mean:.1f}Hz")
        except Exception as e:
            console.print(f"[yellow]   ⚠️ Prosody analysis failed: {e}[/yellow]")
        
        # 3. Audio Emotion
        console.print("[cyan]😊 Detecting audio emotion...[/cyan]")
        try:
            if results["prosody"]:
                results["audio_emotion"] = self._audio_emotion.detect(results["prosody"])
                console.print(f"   Emotion: {results['audio_emotion'].dominant_emotion} ({results['audio_emotion'].confidence:.0%})")
        except Exception as e:
            console.print(f"[yellow]   ⚠️ Audio emotion failed: {e}[/yellow]")
        
        # 4. Facial Emotion (from middle frame)
        console.print("[cyan]😀 Detecting facial emotion...[/cyan]")
        try:
            if self.video_frames:
                mid_frame = self.video_frames[len(self.video_frames) // 2]
                results["facial_emotion"] = self._facial_emotion.detect_from_frame(mid_frame)
                if results["facial_emotion"]:
                    console.print(f"   Emotion: {results['facial_emotion'].dominant_emotion} ({results['facial_emotion'].confidence:.0%})")
        except Exception as e:
            console.print(f"[yellow]   ⚠️ Facial emotion failed: {e}[/yellow]")
        
        # 5. Gesture Detection
        console.print("[cyan]👋 Detecting gestures...[/cyan]")
        try:
            all_gestures = set()
            for frame in self.video_frames[::5]:  # Sample every 5th frame
                gestures = self._gesture.detect(frame)
                all_gestures.update(gestures)
            results["gestures"] = list(all_gestures)
            if results["gestures"]:
                console.print(f"   Gestures: {', '.join(results['gestures'])}")
        except Exception as e:
            console.print(f"[yellow]   ⚠️ Gesture detection failed: {e}[/yellow]")
        
        # Cleanup
        os.unlink(temp_path)

        return results

    def compile_multimodal_intent(self, results: dict) -> tuple[str, dict]:
        """Compile all modalities into a single enriched prompt.

        Returns:
            Tuple of (compiled_intent, detected_signals_dict)
        """
        signals = {
            "emotion": None,
            "emotion_confidence": 0,
            "speaking_style": None,
            "gesture": None
        }

        if not results["text"]:
            return "[No speech detected]", signals

        # Determine overall emotion (prefer facial if confidence is decent)
        if results["facial_emotion"] and results["facial_emotion"].confidence > 0.3:
            signals["emotion"] = results["facial_emotion"].dominant_emotion
            signals["emotion_confidence"] = results["facial_emotion"].confidence
        elif results["audio_emotion"] and results["audio_emotion"].confidence > 0.2:
            signals["emotion"] = results["audio_emotion"].dominant_emotion
            signals["emotion_confidence"] = results["audio_emotion"].confidence

        # Determine speaking style from prosody
        speaking_style_parts = []
        if results["prosody"]:
            p = results["prosody"]
            # Energy-based descriptions
            if p.energy > 0.5:
                speaking_style_parts.append("loud, emphatic")
            elif p.energy > 0.15:
                speaking_style_parts.append("normal")
            else:
                speaking_style_parts.append("quiet, hesitant")

            # Pitch-based descriptions
            if p.pitch_mean > 200:
                speaking_style_parts.append("high-pitched (possibly stressed)")
            elif p.pitch_mean < 120:
                speaking_style_parts.append("low-pitched (calm/serious)")

        signals["speaking_style"] = ", ".join(speaking_style_parts) if speaking_style_parts else None

        # Get gesture
        signals["gesture"] = results["gestures"][0] if results["gestures"] else None

        # Compile using the IntentCompiler - it will naturally integrate the emotion/style
        compiled = self._compiler.compile(
            text=results["text"],
            emotion=signals["emotion"],
            speaking_style=signals["speaking_style"],
            gesture=signals["gesture"]
        )

        return compiled, signals

    def run_interactive(self):
        """Run the interactive demo loop."""
        self._init_components()

        console.print(Panel.fit(
            "[bold green]🎯 Multimodal Intent Fusion - Interactive Demo[/bold green]\n\n"
            "This demo captures:\n"
            "  🎤 Your voice (speech + tone + emotion)\n"
            "  📷 Your face & hands (emotion + gestures)\n\n"
            "And fuses them into a [bold]single enriched prompt[/bold] that helps AI understand you better!",
            border_style="green"
        ))

        while True:
            console.print("\n" + "="*60)
            console.print("[bold]Press ENTER to start recording (or 'q' to quit):[/bold]", end=" ")

            user_input = input()
            if user_input.lower() == 'q':
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break

            # Get duration
            console.print("[dim]Duration in seconds (default 5):[/dim]", end=" ")
            duration_input = input()
            duration = float(duration_input) if duration_input else 5.0

            # Countdown
            console.print("\n[bold yellow]Get ready...[/bold yellow]")
            for i in range(3, 0, -1):
                console.print(f"[bold]{i}...[/bold]")
                time.sleep(1)
            console.print("[bold green]🎬 GO![/bold green]")

            # Record and analyze
            results = self.record_and_analyze(duration)

            # Compile the intent
            console.print("\n[cyan]✨ Compiling multimodal intent...[/cyan]")
            enriched_intent, signals = self.compile_multimodal_intent(results)

            # Display results
            console.print("\n")
            console.print(Panel(
                f"[dim]{results['text']}[/dim]",
                title="📝 Raw Speech",
                border_style="dim"
            ))

            # Summary table - show what was detected
            table = Table(title="🔍 Detected Signals (used to shape the prompt)", show_header=True)
            table.add_column("Signal", style="cyan")
            table.add_column("Detected", style="white")
            table.add_column("Used", style="green")

            # Emotion
            facial_em = f"{results['facial_emotion'].dominant_emotion} ({results['facial_emotion'].confidence:.0%})" if results["facial_emotion"] else "-"
            audio_em = f"{results['audio_emotion'].dominant_emotion} ({results['audio_emotion'].confidence:.0%})" if results["audio_emotion"] else "-"
            used_emotion = f"✓ {signals['emotion']}" if signals['emotion'] else "✗"
            table.add_row("Facial Emotion", facial_em, used_emotion if results["facial_emotion"] and signals["emotion"] == results["facial_emotion"].dominant_emotion else "")
            table.add_row("Audio Emotion", audio_em, used_emotion if results["audio_emotion"] and signals["emotion"] == results["audio_emotion"].dominant_emotion else "")

            # Voice
            if results["prosody"]:
                table.add_row("Voice Energy", f"{results['prosody'].energy:.2f}", "")
                table.add_row("Voice Pitch", f"{results['prosody'].pitch_mean:.0f} Hz", "")
            table.add_row("Speaking Style", signals['speaking_style'] or "-", f"✓ {signals['speaking_style']}" if signals['speaking_style'] else "")

            # Gestures
            if results["gestures"]:
                table.add_row("Gestures", ", ".join(results["gestures"]), f"✓ {signals['gesture']}" if signals['gesture'] else "")

            console.print(table)

            # Show the final compiled intent prominently
            console.print("\n")
            console.print(Panel(
                f"[bold white]{enriched_intent}[/bold white]",
                title="🎯 Compiled Intent (emotion & tone integrated)",
                border_style="green"
            ))

            console.print("\n[dim]👆 Notice how the prompt naturally reflects your emotional state![/dim]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Multimodal Intent Fusion Demo")
    parser.add_argument("--llm-url", default="http://localhost:8082", help="llama.cpp server URL")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration")
    parser.add_argument("--single", action="store_true", help="Run once instead of loop")
    args = parser.parse_args()

    demo = InteractiveMultimodalDemo(llamacpp_url=args.llm_url)

    if args.single:
        demo._init_components()
        results = demo.record_and_analyze(args.duration)
        intent, signals = demo.compile_multimodal_intent(results)
        console.print(f"\n[dim]Detected: emotion={signals['emotion']}, style={signals['speaking_style']}[/dim]")
        console.print(Panel(intent, title="🎯 Compiled Intent", border_style="green"))
    else:
        demo.run_interactive()


if __name__ == "__main__":
    main()

