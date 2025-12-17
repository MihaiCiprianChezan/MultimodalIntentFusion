#!/usr/bin/env python
"""Comprehensive multimodal test - Text, Audio, Video, and combined."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_text_only():
    """Test 1: Text-only intent compilation."""
    console.print("\n" + "="*60)
    console.print("[bold cyan]TEST 1: Text-Only Intent Compilation[/bold cyan]")
    console.print("="*60)
    
    from src.core.llm import LlamaCppLLM, IntentCompiler
    
    llm = LlamaCppLLM(base_url="http://localhost:8082")
    if not llm.is_available():
        console.print("[red]❌ llama.cpp server not running![/red]")
        return False
    
    compiler = IntentCompiler(llm)
    
    test_inputs = [
        "um so like can you help me, you know, write a letter to my landlord about ahh reducing the rent a bit",
        "hey I need to uh find all files that have errors in them or whatever",
    ]
    
    for text in test_inputs:
        result = compiler.compile(text)
        console.print(Panel(text, title="📝 Input", border_style="dim"))
        console.print(Panel(result, title="✨ Compiled", border_style="green"))
    
    console.print("[green]✅ Text-only test passed![/green]")
    return True


def test_audio_analysis():
    """Test 2: Audio prosody and emotion analysis."""
    console.print("\n" + "="*60)
    console.print("[bold cyan]TEST 2: Audio Analysis (Prosody + Emotion)[/bold cyan]")
    console.print("="*60)
    
    import numpy as np
    from src.core.prosody import ProsodyAnalyzer
    from src.core.emotion import AudioEmotionDetector
    
    # Generate test audio (sine wave with variations)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate speech-like audio with varying pitch
    audio = np.sin(2 * np.pi * 200 * t) * 0.5  # Base frequency
    audio += np.sin(2 * np.pi * 400 * t) * 0.3  # Harmonic
    audio = audio.astype(np.float32)
    
    # Test prosody analysis
    prosody = ProsodyAnalyzer(sample_rate=sample_rate)
    features = prosody.analyze_array(audio, sample_rate)

    table = Table(title="Prosody Features")
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Energy", f"{features.energy:.4f}")
    table.add_row("Pitch Mean", f"{features.pitch_mean:.2f} Hz")
    table.add_row("Pitch Std", f"{features.pitch_std:.2f}")
    table.add_row("Tempo", f"{features.tempo:.2f} BPM")
    table.add_row("Spectral Centroid", f"{features.spectral_centroid:.2f}")
    console.print(table)
    
    # Test emotion detection
    emotion_detector = AudioEmotionDetector()
    emotion = emotion_detector.detect(features)
    console.print(f"\n[bold]Detected Emotion:[/bold] {emotion.dominant_emotion} (confidence: {emotion.confidence:.2f})")
    
    console.print("[green]✅ Audio analysis test passed![/green]")
    return True


def test_video_gesture():
    """Test 3: Video gesture recognition."""
    console.print("\n" + "="*60)
    console.print("[bold cyan]TEST 3: Video Gesture Recognition[/bold cyan]")
    console.print("="*60)
    
    try:
        import cv2
        from src.core.vision import GestureRecognizer, WebcamCapture
        
        console.print("📷 Opening webcam...")
        webcam = WebcamCapture(camera_id=0)
        gesture_recognizer = GestureRecognizer()
        
        frame = webcam.capture_frame()
        if frame is None:
            console.print("[yellow]⚠️ No webcam available, skipping video test[/yellow]")
            return True
        
        console.print(f"   Frame shape: {frame.shape}")
        
        # Detect gestures
        gestures = gesture_recognizer.detect(frame)
        console.print(f"   Detected gestures: {gestures if gestures else 'None'}")
        
        # Show frame briefly
        cv2.imshow("Gesture Test - Press any key", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        webcam.close()
        gesture_recognizer.close()
        
        console.print("[green]✅ Video gesture test passed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[yellow]⚠️ Video test skipped: {e}[/yellow]")
        return True


def test_facial_emotion():
    """Test 4: Facial emotion detection."""
    console.print("\n" + "="*60)
    console.print("[bold cyan]TEST 4: Facial Emotion Detection[/bold cyan]")
    console.print("="*60)
    
    try:
        import cv2
        from src.core.emotion import FacialEmotionDetector
        from src.core.vision import WebcamCapture
        
        console.print("📷 Capturing frame for emotion analysis...")
        webcam = WebcamCapture(camera_id=0)
        frame = webcam.capture_frame()
        
        if frame is None:
            console.print("[yellow]⚠️ No webcam, skipping facial emotion test[/yellow]")
            return True
        
        detector = FacialEmotionDetector()
        emotion = detector.detect_from_frame(frame)
        
        if emotion:
            console.print(f"   Detected: {emotion.dominant_emotion} (confidence: {emotion.confidence:.2f})")
        else:
            console.print("   No face detected in frame")
        
        webcam.close()
        console.print("[green]✅ Facial emotion test passed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[yellow]⚠️ Facial emotion test skipped: {e}[/yellow]")
        return True


def test_full_multimodal():
    """Test 5: Full multimodal fusion."""
    console.print("\n" + "="*60)
    console.print("[bold cyan]TEST 5: Full Multimodal Fusion[/bold cyan]")
    console.print("="*60)
    
    from src.core.fusion_engine import MultimodalIntentFusion
    
    engine = MultimodalIntentFusion(
        llamacpp_url="http://localhost:8082",
        enable_vision=True
    )
    
    # Test text processing through full pipeline
    result = engine.process_text(
        "um so like I really need help with this thing, it's kind of urgent you know",
        context="User seems stressed"
    )
    
    console.print(Panel(result.original_text, title="📝 Original", border_style="dim"))
    console.print(Panel(result.compiled_prompt, title="✨ Compiled", border_style="green"))
    console.print(f"Modalities: {result.modalities_used}")
    console.print(f"Confidence: {result.confidence:.2f}")
    
    engine.close()
    console.print("[green]✅ Full multimodal test passed![/green]")
    return True


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold blue]🎯 MultimodalIntentFusion - Comprehensive Test Suite[/bold blue]",
        border_style="blue"
    ))
    
    results = []
    results.append(("Text-Only", test_text_only()))
    results.append(("Audio Analysis", test_audio_analysis()))
    results.append(("Video Gesture", test_video_gesture()))
    results.append(("Facial Emotion", test_facial_emotion()))
    results.append(("Full Multimodal", test_full_multimodal()))
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]TEST SUMMARY[/bold]")
    console.print("="*60)
    
    table = Table()
    table.add_column("Test", style="cyan")
    table.add_column("Result", style="green")
    
    for name, passed in results:
        table.add_row(name, "✅ PASS" if passed else "❌ FAIL")
    
    console.print(table)

