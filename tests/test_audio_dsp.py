"""Unit tests for VoxEmotion Audio DSP module."""

import pytest
import numpy as np

from src.voxemotion.config import AudioDSPConfig
from src.voxemotion.audio_dsp import (
    load_audio,
    extract_features_from_audio,
    extract_ui_telemetry,
    synthesize_emotional_audio,
)


def test_synthesize_emotional_audio():
    for emo in ["happy", "calm", "angry", "sad", "neutral", "disgust"]:
        audio = synthesize_emotional_audio(emo, duration=1.0, sample_rate=22050, seed=42)
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 22050
        assert np.max(np.abs(audio)) <= 1.05
        assert np.min(audio) < 0.0  # Must be AC signal oscillating around zero


def test_extract_features_from_audio():
    cfg = AudioDSPConfig(n_mfcc=40, n_mels=128)
    sr = 22050
    audio = synthesize_emotional_audio("happy", duration=1.5, sample_rate=sr, seed=42)

    features = extract_features_from_audio(audio, sr, config=cfg)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1
    # 40 (MFCC) + 12 (Chroma) + 128 (Mel) = 180
    assert len(features) == 180
    assert not np.isnan(features).any()


def test_extract_ui_telemetry():
    sr = 22050
    audio = synthesize_emotional_audio("angry", duration=1.5, sample_rate=sr, seed=42)
    telemetry = extract_ui_telemetry(audio, sr)

    assert "mean_pitch_hz" in telemetry
    assert "mean_rms_energy" in telemetry
    assert "mean_zcr" in telemetry
    assert "mean_spectral_centroid_hz" in telemetry
    assert "log_mel_spectrogram" in telemetry
    assert telemetry["mean_pitch_hz"] > 0
    assert telemetry["mean_rms_energy"] > 0
