"""Core modules for MultimodalIntentFusion."""

from src.core.fusion_engine import MultimodalIntentFusion
from src.core.models import MultimodalSignal, CompiledIntent, ProsodyFeatures, EmotionResult

__all__ = [
    "MultimodalIntentFusion",
    "MultimodalSignal",
    "CompiledIntent", 
    "ProsodyFeatures",
    "EmotionResult",
]

