"""Unit tests for VoxEmotion Dataset module."""

import pytest
import numpy as np

from src.voxemotion.dataset import (
    parse_ravdess_filename,
    extract_label_from_path,
    build_benchmark_dataset,
)


def test_parse_ravdess_filename():
    # Example: 03-01-03-01-01-01-01.wav -> emotion_code '03' = happy
    meta = parse_ravdess_filename("03-01-03-01-01-01-01.wav")
    assert meta is not None
    assert meta["emotion"] == "happy"
    assert meta["intensity"] == "normal"
    assert meta["actor"] == "01"

    # Example: 03-01-05-02-01-02-12.wav -> emotion_code '05' = angry, intensity 'strong'
    meta2 = parse_ravdess_filename("03-01-05-02-01-02-12.wav")
    assert meta2 is not None
    assert meta2["emotion"] == "angry"
    assert meta2["intensity"] == "strong"

    # Non-matching filename
    assert parse_ravdess_filename("random_sound.wav") is None


def test_extract_label_from_path():
    assert extract_label_from_path("data/sample_happy.wav") == "happy"
    assert extract_label_from_path("data/calm/speech_01.wav") == "calm"
    assert extract_label_from_path("03-01-01-01-01-01-01.wav") == "neutral"


def test_build_benchmark_dataset():
    X, y = build_benchmark_dataset(num_samples_per_emotion=3, observed_emotions=["happy", "sad"])
    assert len(X) == 6
    assert len(y) == 6
    assert X.shape[1] == 180
    assert set(y) == {"happy", "sad"}
