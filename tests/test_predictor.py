"""Unit tests for VoxEmotion Predictor module."""

import pytest
import numpy as np

from src.voxemotion.predictor import EmotionPredictor
from src.voxemotion.audio_dsp import synthesize_emotional_audio


def test_predictor_audio_inference():
    predictor = EmotionPredictor()
    if predictor.model is None:
        pytest.skip("Model artifact not yet trained.")

    audio = synthesize_emotional_audio("happy", duration=1.5, sample_rate=22050, seed=42)
    res = predictor.predict_audio(audio, sample_rate=22050)

    assert "emotion" in res
    assert "confidence" in res
    assert "confidence_percent" in res
    assert "probabilities" in res
    assert "latency_ms" in res
    assert "telemetry" in res

    assert res["emotion"] in predictor.classes
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == len(predictor.classes)
    assert np.isclose(sum(res["probabilities"].values()), 1.0, atol=1e-2)
