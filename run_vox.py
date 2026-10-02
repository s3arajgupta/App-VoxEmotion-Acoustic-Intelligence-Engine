"""VoxEmotion Command-Line Interface (CLI).

Automates acoustic feature extraction, neural network training,
model evaluation, and audio emotion classification.

Usage:
    python run_vox.py --train
    python run_vox.py --evaluate
    python run_vox.py --predict --file samples/sample_happy.wav
    python run_vox.py --generate-samples
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.voxemotion.config import (
    AudioDSPConfig,
    ModelConfig,
    DEFAULT_MODELS_DIR,
    DEFAULT_SAMPLES_DIR,
    DEFAULT_OBSERVED_EMOTIONS,
)
from src.voxemotion.audio_dsp import synthesize_emotional_audio, save_audio_wav
from src.voxemotion.dataset import (
    load_dataset_from_directory,
    build_benchmark_dataset,
)
from src.voxemotion.models import (
    ScaledMLPClassifier,
    save_emotion_bundle,
    load_emotion_bundle,
)
from src.voxemotion.evaluator import evaluate_classifier
from src.voxemotion.predictor import EmotionPredictor


def handle_generate_samples(args):
    """Generate sample audio files across standard emotion classes."""
    print("[INFO] Synthesizing acoustic speech samples...")
    out_dir = Path(args.samples_dir) if args.samples_dir else DEFAULT_SAMPLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, emo in enumerate(DEFAULT_OBSERVED_EMOTIONS):
        y = synthesize_emotional_audio(emo, duration=2.5, sample_rate=22050, seed=42 + i)
        path = save_audio_wav(y, out_dir / f"sample_{emo}.wav", sr=22050)
        print(f"[GENERATED] {emo.upper():<10} -> {path.resolve()}")

    print(f"[SUCCESS] All sample audio files generated in {out_dir.resolve()}")


def handle_train(args):
    """Train neural classifier on audio dataset or benchmark corpus."""
    print("[INFO] Starting VoxEmotion training pipeline...")
    t0 = time.time()
    dsp_cfg = AudioDSPConfig()
    model_cfg = ModelConfig()

    data_dir = Path(args.data_dir) if args.data_dir else None

    if data_dir and data_dir.exists():
        print(f"[INFO] Scanning directory: {data_dir.resolve()} for audio files...")
        X, y, paths = load_dataset_from_directory(data_dir, dsp_config=dsp_cfg)
        print(f"[INFO] Loaded {len(X)} files from disk across {len(set(y))} emotion classes.")
    else:
        print("[INFO] Building acoustic benchmark dataset (180 DSP features)...")
        X, y = build_benchmark_dataset(num_samples_per_emotion=45, dsp_config=dsp_cfg)
        print(f"[INFO] Synthesized {len(X)} samples across {len(set(y))} emotion classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg.test_size, random_state=model_cfg.random_state, stratify=y
    )

    print(f"[INFO] Train Partition: {len(X_train)} samples | Test Partition: {len(X_test)} samples")
    print(f"[INFO] Feature Dimensions: {X.shape[1]} (40 MFCC + 12 Chroma + 128 MelSpectrogram)")

    model = ScaledMLPClassifier(config=model_cfg)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    eval_train = evaluate_classifier(y_train, train_preds, labels=list(model.classes_))
    eval_test = evaluate_classifier(y_test, test_preds, labels=list(model.classes_))

    models_dir = Path(args.models_dir) if args.models_dir else DEFAULT_MODELS_DIR
    save_path = save_emotion_bundle(model, dsp_cfg, save_dir=models_dir)
    elapsed = time.time() - t0

    print("=" * 60)
    print(" VOXEMOTION TRAINING & EVALUATION REPORT")
    print("=" * 60)
    print(f"[METRIC] Train Accuracy:    {eval_train['accuracy']*100:.2f}%")
    print(f"[METRIC] Test Accuracy:     {eval_test['accuracy']*100:.2f}%")
    print(f"[METRIC] Test F1-Macro:     {eval_test['f1_macro']:.4f}")
    print(f"[METRIC] Test F1-Weighted:  {eval_test['f1_weighted']:.4f}")
    print(f"[METRIC] Classes Learned:   {', '.join(model.classes_)}")
    print(f"[SUCCESS] Model artifact saved to: {save_path.resolve()}")
    print(f"[INFO] Pipeline completed in {elapsed:.2f} seconds.")
    print("=" * 60)


def handle_evaluate(args):
    """Evaluate model and output confusion matrix."""
    print("[INFO] Evaluating VoxEmotion model...")
    dsp_cfg = AudioDSPConfig()
    X, y = build_benchmark_dataset(num_samples_per_emotion=30, dsp_config=dsp_cfg)

    predictor = EmotionPredictor()
    if predictor.model is None:
        print("[INFO] Model bundle not found. Training on benchmark dataset first...")
        predictor.train_and_persist(num_samples_per_emotion=40)

    preds = predictor.model.predict(X)
    eval_res = evaluate_classifier(y, preds, labels=predictor.classes)

    print("\n" + "=" * 65)
    print(" VOXEMOTION BENCHMARK EVALUATION MATRIX")
    print("=" * 65)
    print(f"Overall Accuracy:  {eval_res['accuracy']*100:.2f}%")
    print(f"Macro Precision:   {eval_res['precision_macro']:.4f}")
    print(f"Macro Recall:      {eval_res['recall_macro']:.4f}")
    print(f"Macro F1-Score:    {eval_res['f1_macro']:.4f}")
    print("\nConfusion Matrix:")
    print(eval_res["confusion_matrix"].to_string())

    print("\nPer-Emotion Performance Breakdown:")
    rep = eval_res["classification_report"]
    for emo in eval_res["class_names"]:
        if emo in rep:
            f1 = rep[emo]["f1-score"]
            prec = rep[emo]["precision"]
            rec = rep[emo]["recall"]
            print(f"  {emo.capitalize():<12} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("=" * 65)


def handle_predict(args):
    """Predict emotion for a given audio file."""
    if not args.file:
        print("[ERROR] Please provide --file <path_to_wav>")
        sys.exit(1)

    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found at {audio_path.resolve()}")
        sys.exit(1)

    predictor = EmotionPredictor()
    if predictor.model is None:
        print("[INFO] Model bundle not found. Training on benchmark dataset first...")
        predictor.train_and_persist(num_samples_per_emotion=40)

    res = predictor.predict_audio(audio_path)

    print("\n" + "=" * 60)
    print(" VOXEMOTION ACOUSTIC SPEECH INFERENCE")
    print("=" * 60)
    print(f"Audio File:         {audio_path.name}")
    print(f"Duration:           {res['audio_duration_sec']} seconds")
    print(f"Detected Emotion:   {res['emotion'].upper()}")
    print(f"Confidence:         {res['confidence_percent']}%")
    print(f"Acoustic Profile:   {res['description']}")
    print(f"Inference Latency:  {res['latency_ms']:.2f} ms")

    print("\nAcoustic Signal Telemetry:")
    print(f"  Fundamental Pitch (F0): {res['telemetry']['mean_pitch_hz']} Hz")
    print(f"  RMS Signal Energy:      {res['telemetry']['mean_rms_energy']}")
    print(f"  Zero Crossing Rate:     {res['telemetry']['mean_zcr']}")
    print(f"  Spectral Centroid:      {res['telemetry']['mean_spectral_centroid_hz']} Hz")

    print("\nEmotion Probability Distribution:")
    for emo, prob in res["probabilities"].items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len
        print(f"  {emo.capitalize():<10} | {prob*100:>5.1f}% | {bar}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="VoxEmotion: Speech Emotion Recognition & Acoustic Prosody Intelligence Engine"
    )
    parser.add_argument("--train", action="store_true", help="Train MLP neural classifier")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model and show confusion matrix")
    parser.add_argument("--predict", action="store_true", help="Classify emotional tone from audio file")
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample emotional WAV files")

    parser.add_argument("--file", type=str, default=None, help="Path to input audio file (.wav)")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing audio files")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory to save/load models")
    parser.add_argument("--samples-dir", type=str, default=None, help="Directory to output sample audio")

    args = parser.parse_args()

    if args.generate_samples:
        handle_generate_samples(args)
    elif args.train:
        handle_train(args)
    elif args.predict:
        handle_predict(args)
    elif args.evaluate:
        handle_evaluate(args)
    else:
        # Default action: evaluate
        handle_evaluate(args)


if __name__ == "__main__":
    main()
