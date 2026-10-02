"""VoxEmotion: Speech Emotion Recognition & Acoustic Prosody Intelligence Engine.

Digital signal processing (DSP) and neural voice emotion classification package.
"""

from .config import (
    AudioDSPConfig,
    ModelConfig,
    RAVDESS_EMOTIONS,
    DEFAULT_OBSERVED_EMOTIONS,
    EMOTION_META,
    DEFAULT_MODELS_DIR,
    DEFAULT_SAMPLES_DIR,
)
from .audio_dsp import (
    load_audio,
    extract_features_from_audio,
    extract_ui_telemetry,
    synthesize_emotional_audio,
    save_audio_wav,
)
from .dataset import (
    parse_ravdess_filename,
    load_dataset_from_directory,
    build_benchmark_dataset,
)
from .models import (
    ScaledMLPClassifier,
    ScaledSVMClassifier,
    save_emotion_bundle,
    load_emotion_bundle,
)
from .evaluator import evaluate_classifier
from .predictor import EmotionPredictor

__version__ = "2.0.0"

__all__ = [
    "AudioDSPConfig",
    "ModelConfig",
    "RAVDESS_EMOTIONS",
    "DEFAULT_OBSERVED_EMOTIONS",
    "EMOTION_META",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_SAMPLES_DIR",
    "load_audio",
    "extract_features_from_audio",
    "extract_ui_telemetry",
    "synthesize_emotional_audio",
    "save_audio_wav",
    "parse_ravdess_filename",
    "load_dataset_from_directory",
    "build_benchmark_dataset",
    "ScaledMLPClassifier",
    "ScaledSVMClassifier",
    "save_emotion_bundle",
    "load_emotion_bundle",
    "evaluate_classifier",
    "EmotionPredictor",
]
