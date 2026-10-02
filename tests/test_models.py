"""Unit tests for VoxEmotion Neural Classifier models."""

import pytest
import numpy as np
from pathlib import Path

from src.voxemotion.config import AudioDSPConfig, ModelConfig
from src.voxemotion.models import (
    ScaledMLPClassifier,
    ScaledSVMClassifier,
    save_emotion_bundle,
    load_emotion_bundle,
)


@pytest.fixture
def synthetic_speech_features():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 180)
    y = np.array(["happy"] * 20 + ["angry"] * 20 + ["calm"] * 20)
    return X, y


def test_scaled_mlp_classifier(synthetic_speech_features):
    X, y = synthetic_speech_features
    cfg = ModelConfig(max_iter=30, batch_size=16)
    mlp = ScaledMLPClassifier(config=cfg)
    mlp.fit(X, y)

    preds = mlp.predict(X)
    probs = mlp.predict_proba(X)

    assert len(preds) == len(y)
    assert probs.shape == (len(y), 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert set(mlp.classes_) == {"angry", "calm", "happy"}


def test_scaled_svm_classifier(synthetic_speech_features):
    X, y = synthetic_speech_features
    svm = ScaledSVMClassifier(C=1.0)
    svm.fit(X, y)

    preds = svm.predict(X)
    assert len(preds) == len(y)


def test_model_bundle_persistence(synthetic_speech_features, tmp_path):
    X, y = synthetic_speech_features
    cfg = ModelConfig(max_iter=20)
    dsp_cfg = AudioDSPConfig()

    mlp = ScaledMLPClassifier(config=cfg)
    mlp.fit(X, y)

    save_path = save_emotion_bundle(mlp, dsp_cfg, save_dir=tmp_path)
    assert save_path.exists()

    loaded_mlp, loaded_dsp, loaded_classes = load_emotion_bundle(save_dir=tmp_path)
    assert loaded_classes == list(mlp.classes_)

    p_orig = mlp.predict(X[:5])
    p_load = loaded_mlp.predict(X[:5])
    assert list(p_orig) == list(p_load)
