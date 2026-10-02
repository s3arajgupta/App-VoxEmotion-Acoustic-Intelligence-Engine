"""VoxEmotion Dataset and Feature Loader Module.

Parses RAVDESS audio datasets, scans acoustic directories, and generates synthetic
benchmark corpora for offline evaluation and testing.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    AudioDSPConfig,
    RAVDESS_EMOTIONS,
    DEFAULT_OBSERVED_EMOTIONS,
    DEFAULT_DATA_DIR,
)
from .audio_dsp import load_audio, extract_features_from_audio, synthesize_emotional_audio


def parse_ravdess_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse standard 7-part RAVDESS audio filename metadata.

    Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
    Example: 03-01-03-01-01-01-01.wav (happy, normal intensity, actor 01)
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) == 7:
        emotion_code = parts[2]
        emotion = RAVDESS_EMOTIONS.get(emotion_code, "unknown")
        return {
            "modality": parts[0],
            "vocal_channel": parts[1],
            "emotion_code": emotion_code,
            "emotion": emotion,
            "intensity": "normal" if parts[3] == "01" else "strong",
            "statement": parts[4],
            "repetition": parts[5],
            "actor": parts[6],
        }
    return None


def extract_label_from_path(path: Union[str, Path]) -> Optional[str]:
    """Infer emotion label from RAVDESS filename, stem name, or parent directory."""
    path_obj = Path(path)
    stem = path_obj.stem.lower()

    # 1. Check RAVDESS convention
    ravdess_meta = parse_ravdess_filename(stem)
    if ravdess_meta and ravdess_meta["emotion"] in DEFAULT_OBSERVED_EMOTIONS:
        return ravdess_meta["emotion"]

    # 2. Check stem substrings (e.g. sample_happy, angry_01)
    for emo in DEFAULT_OBSERVED_EMOTIONS:
        if emo in stem:
            return emo

    # 3. Check parent directory name
    parent_name = path_obj.parent.name.lower()
    for emo in DEFAULT_OBSERVED_EMOTIONS:
        if emo in parent_name:
            return emo

    return None


def load_dataset_from_directory(
    dir_path: Union[str, Path],
    observed_emotions: Optional[List[str]] = None,
    dsp_config: Optional[AudioDSPConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Scan directory recursively for WAV audio files and extract acoustic feature vectors.

    Args:
        dir_path: Root folder containing audio files.
        observed_emotions: List of target emotions to include.
        dsp_config: Optional AudioDSPConfig.

    Returns:
        Tuple of (Feature matrix X, Labels vector y, List of filepaths processed).
    """
    root_path = Path(dir_path)
    target_emotions = set(observed_emotions or DEFAULT_OBSERVED_EMOTIONS)
    cfg = dsp_config or AudioDSPConfig()

    data, labels, paths = [], [], []

    for file_path in root_path.rglob("*.wav"):
        label = extract_label_from_path(file_path)
        if label and label in target_emotions:
            try:
                y, sr = load_audio(file_path, target_sr=cfg.sample_rate)
                feat = extract_features_from_audio(y, sr, cfg)
                data.append(feat)
                labels.append(label)
                paths.append(str(file_path))
            except Exception as e:
                continue

    if not data:
        return np.empty((0, cfg.total_features_count)), np.empty(0), []

    return np.asarray(data, dtype=np.float64), np.asarray(labels), paths


def build_benchmark_dataset(
    num_samples_per_emotion: int = 30,
    observed_emotions: Optional[List[str]] = None,
    dsp_config: Optional[AudioDSPConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate high-fidelity acoustic corpus with realistic emotional vocal prosody.

    Enables reliable out-of-the-box model training and evaluation without external multi-GB downloads.

    Args:
        num_samples_per_emotion: Number of distinct audio samples synthesized per emotion.
        observed_emotions: Target emotions to generate.
        dsp_config: Optional AudioDSPConfig.

    Returns:
        Tuple of (Feature matrix X, Label array y).
    """
    target_emotions = observed_emotions or DEFAULT_OBSERVED_EMOTIONS
    cfg = dsp_config or AudioDSPConfig()

    X_list, y_list = [], []

    for emo_idx, emotion in enumerate(target_emotions):
        for s in range(num_samples_per_emotion):
            seed = 1000 * (emo_idx + 1) + s
            # Vary duration and prosody
            dur = 2.0 + (s % 3) * 0.5
            audio = synthesize_emotional_audio(emotion, duration=dur, sample_rate=cfg.sample_rate, seed=seed)
            feat = extract_features_from_audio(audio, cfg.sample_rate, cfg)
            X_list.append(feat)
            y_list.append(emotion)

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list)
    return X, y
