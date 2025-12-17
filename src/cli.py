"""Command-line interface for MultimodalIntentFusion."""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """MultimodalIntentFusion - Transform natural speech into precise AI instructions."""
    pass


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--context", "-c", default="", help="Additional context for compilation")
@click.option("--whisper-model", "-w", default="base.en", help="Whisper model size")
@click.option("--llm-model", "-l", default="llama3.1:8b", help="Ollama model name (fallback)")
@click.option("--llm-url", "-u", default="http://localhost:8082", help="llama.cpp server URL")
def process(audio_file: str, context: str, whisper_model: str, llm_model: str, llm_url: str):
    """Process an audio file through the intent fusion pipeline."""
    from src.core.fusion_engine import MultimodalIntentFusion

    console.print(f"\n[bold blue]🎯 MultimodalIntentFusion[/bold blue]")
    console.print(f"Processing: {audio_file}\n")

    # Initialize engine
    engine = MultimodalIntentFusion(
        whisper_model=whisper_model,
        llm_model=llm_model,
        llamacpp_url=llm_url
    )

    # Process audio
    result = engine.process_audio(audio_file, context=context)

    # Display results
    _display_result(result)


@cli.command()
@click.option("--duration", "-d", default=5.0, help="Recording duration in seconds")
@click.option("--context", "-c", default="", help="Additional context")
@click.option("--vision/--no-vision", default=False, help="Enable webcam capture")
@click.option("--llm-url", "-u", default="http://localhost:8082", help="llama.cpp server URL")
def record(duration: float, context: str, vision: bool, llm_url: str):
    """Record from microphone and process."""
    from src.core.fusion_engine import MultimodalIntentFusion

    console.print(f"\n[bold blue]🎯 MultimodalIntentFusion[/bold blue]")
    console.print(f"Recording for {duration} seconds...\n")

    engine = MultimodalIntentFusion(enable_vision=vision, llamacpp_url=llm_url)

    try:
        result = engine.record_and_process(
            duration=duration,
            capture_video=vision,
            context=context
        )
        _display_result(result)
    finally:
        engine.close()


@cli.command()
@click.argument("text")
@click.option("--context", "-c", default="", help="Additional context")
@click.option("--llm-url", "-u", default="http://localhost:8082", help="llama.cpp server URL")
def compile(text: str, context: str, llm_url: str):
    """Compile text directly into clear intent (MVP mode)."""
    from src.core.fusion_engine import MultimodalIntentFusion

    console.print(f"\n[bold blue]🎯 MultimodalIntentFusion - Text Mode[/bold blue]\n")

    engine = MultimodalIntentFusion(llamacpp_url=llm_url)
    result = engine.process_text(text, context=context)

    _display_result(result)


@cli.command()
@click.option("--llm-url", "-u", default="http://localhost:8082", help="llama.cpp server URL")
def interactive(llm_url: str):
    """Start interactive mode - continuous recording and processing."""
    from src.core.fusion_engine import MultimodalIntentFusion

    console.print(f"\n[bold blue]🎯 MultimodalIntentFusion - Interactive Mode[/bold blue]")
    console.print("Press Enter to start recording, Ctrl+C to exit\n")

    engine = MultimodalIntentFusion(llamacpp_url=llm_url)
    
    try:
        while True:
            input("Press Enter to record (5 seconds)...")
            result = engine.record_and_process(duration=5.0)
            _display_result(result)
            console.print("\n" + "="*50 + "\n")
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    finally:
        engine.close()


@cli.command()
@click.option("--llm-url", "-u", default="http://localhost:8082", help="llama.cpp server URL")
def setup(llm_url: str):
    """Check available LLM backends (llama.cpp and Ollama)."""
    from src.core.llm import LlamaCppLLM, OllamaLLM

    console.print(f"\n[bold blue]🔧 LLM Backend Check[/bold blue]\n")

    # Check llama.cpp first
    llamacpp = LlamaCppLLM(base_url=llm_url)
    if llamacpp.is_available():
        console.print(f"[green]✅ llama.cpp server running at {llm_url}[/green]")
        model_info = llamacpp.get_model_info()
        if model_info:
            console.print(f"   Model info: {model_info}")
    else:
        console.print(f"[yellow]⚠️ llama.cpp not available at {llm_url}[/yellow]")
        console.print("   Start with: llama-server.exe --model <path> --port 8082")

    # Check Ollama
    console.print()
    try:
        ollama = OllamaLLM()
        models = ollama.list_models()
        if models:
            console.print("[green]✅ Ollama is running[/green]")
            console.print(f"   Available models: {', '.join(models)}")
        else:
            console.print("[yellow]⚠️ Ollama running but no models found[/yellow]")
    except ImportError:
        console.print("[dim]ℹ️ Ollama not installed (optional)[/dim]")
    except Exception:
        console.print("[dim]ℹ️ Ollama not running (optional)[/dim]")

    console.print("\n[green]Setup check complete![/green]")


@cli.command()
@click.option("--llm-url", "-u", default="http://localhost:8082", help="llama.cpp server URL")
def test_llm(llm_url: str):
    """Quick test of the LLM connection."""
    from src.core.llm import LlamaCppLLM

    console.print(f"\n[bold blue]🧪 Testing LLM at {llm_url}[/bold blue]\n")

    llm = LlamaCppLLM(base_url=llm_url)

    if not llm.is_available():
        console.print(f"[red]❌ Cannot connect to {llm_url}[/red]")
        console.print("\nStart llama.cpp server with:")
        console.print("[dim]llama-server.exe --model <path.gguf> --port 8082[/dim]")
        return

    console.print("[green]✅ Connected![/green]")
    console.print("\nTesting generation...")

    try:
        response = llm.generate(
            "Say 'Hello, I am working!' in exactly those words.",
            system_prompt="You are a helpful assistant. Be very brief."
        )
        console.print(f"\n[cyan]Response:[/cyan] {response}")
        console.print("\n[green]✅ LLM is working correctly![/green]")
    except Exception as e:
        console.print(f"[red]❌ Generation failed: {e}[/red]")


def _display_result(result):
    """Display compiled intent result."""
    # Original text
    console.print(Panel(
        result.original_text or "[No text]",
        title="📝 Original Input",
        border_style="dim"
    ))
    
    # Compiled intent
    console.print(Panel(
        result.compiled_prompt,
        title="✨ Compiled Intent",
        border_style="green"
    ))
    
    # Metadata table
    table = Table(title="📊 Analysis", show_header=True)
    table.add_column("Modality", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Modalities Used", ", ".join(result.modalities_used))
    table.add_row("Confidence", f"{result.confidence:.0%}")
    
    if result.emotion_context:
        table.add_row("Emotion", result.emotion_context)
    
    if result.metadata.get("gesture"):
        table.add_row("Gesture", result.metadata["gesture"])
    
    console.print(table)


if __name__ == "__main__":
    cli()

