#!/usr/bin/env python3
"""
MultimodalIntentFusion Demo Script

Demonstrates the core functionality of the intent fusion engine.
Run with: python examples/demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

console = Console()


def demo_text_compilation():
    """Demo: Text-only intent compilation (MVP mode)."""
    from src.core.fusion_engine import MultimodalIntentFusion
    
    console.print("\n[bold cyan]═══ Demo 1: Text Intent Compilation ═══[/bold cyan]\n")
    
    engine = MultimodalIntentFusion()
    
    # Example messy inputs
    test_inputs = [
        "um so like I need you to, you know, help me write an email to my boss about, uh, taking Friday off",
        "hey can you maybe like look at this code and tell me why it's not working or whatever",
        "I want to, hmm, let me think... create a presentation about AI stuff for next week's meeting",
    ]
    
    for i, text in enumerate(test_inputs, 1):
        console.print(f"[yellow]Input {i}:[/yellow] {text}\n")
        
        result = engine.process_text(text)
        
        console.print(Panel(
            result.compiled_prompt,
            title="✨ Compiled Intent",
            border_style="green"
        ))
        console.print()


def demo_prosody_analysis():
    """Demo: Prosody feature extraction."""
    from src.core.prosody import ProsodyAnalyzer
    from src.core.emotion import AudioEmotionDetector
    import numpy as np
    
    console.print("\n[bold cyan]═══ Demo 2: Prosody Analysis ═══[/bold cyan]\n")
    
    analyzer = ProsodyAnalyzer()
    emotion_detector = AudioEmotionDetector()
    
    # Generate synthetic audio for demo (sine wave)
    console.print("[dim]Generating synthetic audio for demo...[/dim]\n")
    
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Different "emotional" patterns
    patterns = {
        "calm": np.sin(2 * np.pi * 200 * t) * 0.3,
        "energetic": np.sin(2 * np.pi * 400 * t) * 0.8 + np.random.randn(len(t)) * 0.1,
        "variable": np.sin(2 * np.pi * (200 + 100 * np.sin(2 * np.pi * 2 * t)) * t) * 0.5,
    }
    
    for name, audio in patterns.items():
        features = analyzer.analyze_array(audio.astype(np.float32), sample_rate)
        emotion = emotion_detector.detect(features)
        
        console.print(f"[cyan]{name.upper()}[/cyan]")
        console.print(f"  Energy: {features.energy:.4f} ({features.get_energy_description()})")
        console.print(f"  Tempo: {features.tempo:.1f} ({features.get_tempo_description()})")
        console.print(f"  Detected emotion: {emotion.dominant_emotion} ({emotion.confidence:.0%})")
        console.print()


def demo_emotion_detection():
    """Demo: Audio emotion detection from prosody."""
    from src.core.models import ProsodyFeatures
    from src.core.emotion import AudioEmotionDetector
    
    console.print("\n[bold cyan]═══ Demo 3: Emotion Detection ═══[/bold cyan]\n")
    
    detector = AudioEmotionDetector()
    
    # Simulated prosody patterns
    patterns = [
        ("Calm speaker", ProsodyFeatures(energy=0.04, pitch_std=15, tempo=100)),
        ("Excited speaker", ProsodyFeatures(energy=0.12, pitch_std=40, tempo=140)),
        ("Sad speaker", ProsodyFeatures(energy=0.02, pitch_std=8, tempo=75)),
        ("Angry speaker", ProsodyFeatures(energy=0.18, pitch_std=50, tempo=160)),
    ]
    
    for name, prosody in patterns:
        result = detector.detect(prosody)
        
        console.print(f"[cyan]{name}[/cyan]")
        console.print(f"  Prosody: energy={prosody.energy:.2f}, pitch_var={prosody.pitch_std:.0f}, tempo={prosody.tempo:.0f}")
        console.print(f"  Result: {result.dominant_emotion} ({result.confidence:.0%})")
        
        # Show all probabilities
        top_emotions = sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        probs_str = ", ".join(f"{e}: {p:.0%}" for e, p in top_emotions)
        console.print(f"  Probabilities: {probs_str}")
        console.print()


def main():
    """Run all demos."""
    console.print(Panel(
        "[bold]MultimodalIntentFusion Demo[/bold]\n\n"
        "Transform natural human communication into precise AI instructions.",
        title="🎯 Welcome",
        border_style="blue"
    ))
    
    try:
        demo_text_compilation()
    except Exception as e:
        console.print(f"[red]Text compilation demo failed: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running with a model installed.[/yellow]")
    
    demo_prosody_analysis()
    demo_emotion_detection()
    
    console.print(Panel(
        "To use the full pipeline with audio:\n"
        "  python main.py process <audio_file.wav>\n"
        "  python main.py record --duration 5\n"
        "  python main.py compile \"your text here\"",
        title="📖 Next Steps",
        border_style="green"
    ))


if __name__ == "__main__":
    main()

