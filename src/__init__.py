"""
MultimodalIntentFusion - Transform natural human communication into precise AI instructions.

A multimodal intent compilation engine that fuses speech, prosody, emotion, and visual
signals to create clear, machine-ready prompts.
"""

__version__ = "0.1.0"
__author__ = "MultimodalIntentFusion Team"

from src.core.fusion_engine import MultimodalIntentFusion
from src.core.models import MultimodalSignal, CompiledIntent

__all__ = [
    "MultimodalIntentFusion",
    "MultimodalSignal", 
    "CompiledIntent",
]

