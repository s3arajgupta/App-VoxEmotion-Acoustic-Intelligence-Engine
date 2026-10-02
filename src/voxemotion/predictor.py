"""VoxEmotion Real-Time Inference Engine.

Predicts vocal emotion from raw audio files, memory buffers, or numpy waveforms,
returning probability distributions, confidence scores, and acoustic telemetry.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import time
import io
import numpy as np

from .config import (
    AudioDSPConfig,
    ModelConfig,
    EMOTION_META,
    DEFAULT_MODELS_DIR,
)
from .audio_dsp import load_audio, extract_features_from_audio, extract_ui_telemetry
from .models import (
    ScaledMLPClassifier,
    load_emotion_bundle,
    save_emotion_bundle,
)
from .dataset import build_benchmark_dataset


class EmotionPredictor:
    """Production inference engine for Speech Emotion Recognition."""

    def __init__(self, models_dir: Optional[Path] = None, auto_load: bool = True):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.model: Optional[ScaledMLPClassifier] = None
        self.dsp_config = AudioDSPConfig()
        self.classes: List[str] = []

        if auto_load:
            try:
                self.load()
            except Exception:
                pass

    def load(self) -> None:
        """Load trained model bundle from disk."""
        self.model, self.dsp_config, self.classes = load_emotion_bundle(self.models_dir)

    def train_and_persist(self, num_samples_per_emotion: int = 40) -> Dict[str, Any]:
        """Train classifier on acoustic benchmark dataset and serialize bundle."""
        X, y = build_benchmark_dataset(num_samples_per_emotion=num_samples_per_emotion, dsp_config=self.dsp_config)
        self.model = ScaledMLPClassifier()
        self.model.fit(X, y)
        self.classes = list(self.model.classes_)
        bundle_path = save_emotion_bundle(self.model, self.dsp_config, self.models_dir)
        return {
            "saved_to": str(bundle_path),
            "classes": self.classes,
            "samples_trained": len(X),
        }

    def predict_audio(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Classify emotional state from audio signal.

        Args:
            audio_input: Filepath, byte buffer, or 1D audio array.
            sample_rate: Sample rate if numpy array is passed.

        Returns:
            Dictionary containing predicted emotion, confidence, probability distribution,
            latency, and acoustic telemetry.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Train or load model bundle first.")

        t0 = time.perf_counter()

        # Load / normalize audio
        if isinstance(audio_input, np.ndarray):
            y = audio_input.astype(np.float32)
            sr = sample_rate or self.dsp_config.sample_rate
        else:
            y, sr = load_audio(audio_input, target_sr=self.dsp_config.sample_rate)

        # Extract features
        features = extract_features_from_audio(y, sr, self.dsp_config)

        # Inference
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        top_idx = int(np.argmax(probs))
        predicted_emotion = str(self.classes[top_idx])
        confidence = float(probs[top_idx])

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Construct probability mapping
        prob_dist = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.classes, probs)}
        # Sort descending by probability
        prob_dist = dict(sorted(prob_dist.items(), key=lambda item: item[1], reverse=True))

        meta = EMOTION_META.get(predicted_emotion, {"emoji": "🎙️", "color": "#3B82F6", "desc": "Speech tone"})
        telemetry = extract_ui_telemetry(y, sr)

        return {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100.0, 1),
            "emoji": meta["emoji"],
            "color": meta["color"],
            "description": meta["desc"],
            "probabilities": prob_dist,
            "latency_ms": round(latency_ms, 2),
            "telemetry": telemetry,
            "sample_rate": sr,
            "audio_duration_sec": round(len(y) / sr, 2),
        }
