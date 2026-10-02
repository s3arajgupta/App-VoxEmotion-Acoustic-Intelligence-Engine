"""VoxEmotion Machine Learning Architectures.

Provides Scaled MLP neural network, Support Vector Machine (SVC) acoustic baseline,
and persistence utilities for model bundles.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig, AudioDSPConfig, DEFAULT_MODELS_DIR


class ScaledMLPClassifier:
    """Standardized Multi-Layer Perceptron neural network for Speech Emotion Recognition."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            alpha=self.config.alpha,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            learning_rate_init=self.config.learning_rate_init,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.15,
        )
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledMLPClassifier":
        """Scale acoustic features and train multilayer perceptron."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict emotion labels."""
        X = np.atleast_2d(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict normalized emotion probability distribution."""
        X = np.atleast_2d(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    @property
    def loss_curve_(self) -> List[float]:
        return list(self.model.loss_curve_) if hasattr(self.model, "loss_curve_") else []


class ScaledSVMClassifier:
    """Support Vector Machine (SVC with RBF kernel) acoustic baseline."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = SVC(C=C, kernel="rbf", probability=True, random_state=random_state)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledSVMClassifier":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.model.predict_proba(self.scaler.transform(X))


def save_emotion_bundle(
    model: ScaledMLPClassifier,
    dsp_config: AudioDSPConfig,
    save_dir: Optional[Path] = None,
) -> Path:
    """Serialize model artifact, scaler, classes, and DSP config."""
    out_dir = Path(save_dir) if save_dir else DEFAULT_MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = out_dir / "vox_emotion_bundle.pkl"
    payload = {
        "model": model,
        "classes": list(model.classes_),
        "dsp_config": dsp_config,
        "loss_curve": model.loss_curve_,
    }
    joblib.dump(payload, bundle_path)
    return bundle_path


def load_emotion_bundle(save_dir: Optional[Path] = None) -> Tuple[ScaledMLPClassifier, AudioDSPConfig, List[str]]:
    """Load serialized emotion model bundle."""
    target_dir = Path(save_dir) if save_dir else DEFAULT_MODELS_DIR
    bundle_path = target_dir / "vox_emotion_bundle.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {bundle_path.resolve()}")

    payload = joblib.load(bundle_path)
    return payload["model"], payload["dsp_config"], payload["classes"]
