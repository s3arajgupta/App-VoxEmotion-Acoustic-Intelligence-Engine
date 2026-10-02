"""VoxEmotion Configuration Module.

Centralizes acoustic digital signal processing (DSP) configurations, RAVDESS emotion codes,
color palettes, and neural classifier hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "samples"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Standard RAVDESS Emotion Numerical Codes
RAVDESS_EMOTIONS: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Observed Emotions in Standard Benchmark Subset
DEFAULT_OBSERVED_EMOTIONS: List[str] = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "disgust",
]

# UI Metadata and Visual Color Tokens
EMOTION_META: Dict[str, Dict[str, str]] = {
    "happy": {"emoji": "😄", "color": "#10B981", "desc": "High pitch variance, elevated energy, bright timbre"},
    "calm": {"emoji": "😌", "color": "#06B6D4", "desc": "Smooth fundamental frequency, steady low cadence"},
    "neutral": {"emoji": "😐", "color": "#64748B", "desc": "Balanced acoustic prosody, standard conversational cadence"},
    "sad": {"emoji": "😢", "color": "#3B82F6", "desc": "Depressed pitch contour, low energy, softened formants"},
    "angry": {"emoji": "😡", "color": "#EF4444", "desc": "Harsh spectral energy, high acoustic power, sharp attack"},
    "fearful": {"emoji": "😨", "color": "#8B5CF6", "desc": "Tremor in harmonic frequencies, erratic pitch modulation"},
    "disgust": {"emoji": "🤢", "color": "#D97706", "desc": "Guttural formant shifts, downward intonation contour"},
    "surprised": {"emoji": "😲", "color": "#EC4899", "desc": "Sudden upward pitch rise, prominent high-band spectral energy"},
}


@dataclass
class AudioDSPConfig:
    """Acoustic Digital Signal Processing (DSP) parameters."""
    sample_rate: int = 22050
    duration_seconds: float = 3.0
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 8000.0

    # Feature Dimensions: MFCC (40) + Chroma (12) + MelSpectrogram (128) = 180
    @property
    def total_features_count(self) -> int:
        return self.n_mfcc + 12 + self.n_mels


@dataclass
class ModelConfig:
    """Classifier hyperparameters."""
    hidden_layer_sizes: Tuple[int, ...] = (256, 128)
    alpha: float = 0.01  # L2 regularization
    batch_size: int = 32
    learning_rate: str = "adaptive"
    learning_rate_init: float = 0.001
    max_iter: int = 350
    random_state: int = 42
    test_size: float = 0.20
    models_dir: Path = DEFAULT_MODELS_DIR
