"""Unit tests for VoxEmotion Evaluator module."""

import pytest
import numpy as np

from src.voxemotion.evaluator import evaluate_classifier


def test_evaluate_classifier():
    y_true = np.array(["happy", "angry", "calm", "happy", "angry", "calm"])
    y_pred = np.array(["happy", "angry", "calm", "happy", "calm", "calm"])

    res = evaluate_classifier(y_true, y_pred)

    assert "accuracy" in res
    assert "f1_macro" in res
    assert "confusion_matrix" in res
    assert res["accuracy"] == pytest.approx(5 / 6, rel=1e-2)
    assert res["confusion_matrix"].shape == (3, 3)
