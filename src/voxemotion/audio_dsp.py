"""VoxEmotion Acoustic Digital Signal Processing (DSP) Module.

Provides robust audio waveform loading, frequency-domain representations,
MFCC/Chroma/Mel-Spectrogram feature extraction, and synthetic emotional voice prosody generation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import io
import numpy as np
import soundfile as sf
import librosa

from .config import AudioDSPConfig


def load_audio(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = 22050,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Load and normalize an audio signal to mono floating-point format.

    Args:
        file_or_path: Filepath, raw bytes, or BytesIO buffer.
        target_sr: Target sampling rate (defaults to 22,050 Hz).
        duration: Optional maximum duration in seconds.

    Returns:
        Tuple of (1D float32 audio waveform, sampling rate).
    """
    if isinstance(file_or_path, bytes):
        file_or_path = io.BytesIO(file_or_path)

    try:
        # Load using soundfile directly if possible
        with sf.SoundFile(file_or_path) as sfile:
            y = sfile.read(dtype="float32")
            sr = sfile.samplerate

            # Convert stereo to mono if needed
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            # Resample if sample rate does not match
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
    except Exception:
        # Fallback to librosa.load
        y, sr = librosa.load(file_or_path, sr=target_sr, mono=True, duration=duration)

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Pad or trim to target duration if specified
    if duration is not None:
        target_samples = int(target_sr * duration)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), mode="constant")
        else:
            y = y[:target_samples]

    # Peak normalization to prevent clipping
    max_amp = np.max(np.abs(y))
    if max_amp > 1e-6:
        y = y / max_amp

    return y.astype(np.float32), sr


def extract_features_from_audio(
    y: np.ndarray,
    sr: int,
    config: Optional[AudioDSPConfig] = None,
) -> np.ndarray:
    """Extract standard 180-dimensional acoustic feature vector.

    Components:
    - 40 MFCCs (Mel-Frequency Cepstral Coefficients)
    - 12 Chroma STFT pitch class energies
    - 128 Mel-scale Spectrogram band energies

    Args:
        y: 1D audio waveform.
        sr: Audio sample rate.
        config: Optional AudioDSPConfig.

    Returns:
        1D float64 feature vector of length 180.
    """
    cfg = config or AudioDSPConfig()

    # Ensure non-empty signal
    if len(y) < cfg.n_fft:
        y = np.pad(y, (0, cfg.n_fft - len(y)), mode="constant")

    # 1. Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    mfcc_mean = np.mean(mfccs.T, axis=0)

    # 2. Chroma STFT (12 semitone pitch classes)
    stft = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, hop_length=cfg.hop_length)
    chroma_mean = np.mean(chroma.T, axis=0)

    # 3. Mel-scale Spectrogram (128 mel frequency bands)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    mel_mean = np.mean(mel.T, axis=0)

    # Concatenate features: 40 + 12 + 128 = 180 dims
    features = np.hstack((mfcc_mean, chroma_mean, mel_mean))
    return np.asarray(features, dtype=np.float64)


def extract_ui_telemetry(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute rich visual telemetry for UI waveform and spectrogram rendering."""
    # Compute RMS Energy envelope
    rms = librosa.feature.rms(y=y, hop_length=512)[0]

    # Compute Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=512)[0]

    # Compute Spectral Centroid (timbral brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]

    # Log-Mel Spectrogram for heatmap display
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Pitch estimation (fundamental frequency f0 track)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)]
        mean_pitch = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 150.0
    except Exception:
        mean_pitch = 150.0

    return {
        "mean_pitch_hz": round(mean_pitch, 1),
        "mean_rms_energy": round(float(np.mean(rms)), 4),
        "mean_zcr": round(float(np.mean(zcr)), 4),
        "mean_spectral_centroid_hz": round(float(np.mean(centroid)), 1),
        "duration_sec": round(float(len(y) / sr), 2),
        "rms_envelope": rms,
        "log_mel_spectrogram": log_mel,
    }


def synthesize_emotional_audio(
    emotion: str,
    duration: float = 2.5,
    sample_rate: int = 22050,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Synthesize acoustic voice signal with emotional prosody characteristics.

    Simulates speech formants, pitch modulation (jitter), and harmonic overtones.

    Args:
        emotion: Target emotion ('neutral', 'calm', 'happy', 'sad', 'angry', 'disgust', etc.).
        duration: Audio duration in seconds.
        sample_rate: Target sample rate.
        seed: Optional RNG seed.

    Returns:
        1D float32 audio signal.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    emo = emotion.strip().lower()

    if emo == "angry":
        # High F0, energetic vibrato, heavy harmonic distortion, abrupt attack
        f0 = 240 + 35 * np.sin(2 * np.pi * 5.5 * t) + rng.randn(n_samples) * 3
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        signal = (
            0.55 * np.sin(phase)
            + 0.30 * np.sin(2 * phase)
            + 0.20 * np.sin(3 * phase)
            + 0.15 * np.sin(4 * phase)
        )
        env = (np.sin(np.pi * t / duration) ** 0.5) * (1.0 + 0.15 * rng.randn(n_samples))
        signal = signal * env

    elif emo == "happy":
        # High rising and falling pitch contour, bright harmonics, active cadence
        f0 = 210 + 65 * np.sin(2 * np.pi * 2.8 * t) + 15 * np.sin(2 * np.pi * 0.8 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        signal = (
            0.50 * np.sin(phase)
            + 0.35 * np.sin(2 * phase)
            + 0.18 * np.sin(3 * phase)
        )
        env = np.sin(np.pi * t / duration) ** 0.75
        signal = signal * env

    elif emo == "calm":
        # Low, gentle F0, smooth breathing envelope, soft pure tones
        f0 = 125 + 8 * np.sin(2 * np.pi * 0.8 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        signal = 0.75 * np.sin(phase) + 0.20 * np.sin(2 * phase)
        env = np.sin(np.pi * t / duration) ** 1.3
        signal = signal * env

    elif emo == "sad":
        # Falling pitch trajectory, low energy, muted harmonics, tremolo
        f0 = np.linspace(165, 110, n_samples) + 4 * np.sin(2 * np.pi * 3.0 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        signal = 0.70 * np.sin(phase) + 0.15 * np.sin(2 * phase)
        env = (np.sin(np.pi * t / duration) ** 1.6) * 0.45
        signal = signal * env

    elif emo == "disgust":
        # Low guttural F0, vocal fry emulation, uneven subharmonics
        f0 = 110 + 20 * np.sin(2 * np.pi * 1.2 * t) + rng.randn(n_samples) * 5
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        signal = 0.50 * np.sin(phase) + 0.30 * np.sin(1.5 * phase) + 0.25 * np.sin(2.5 * phase)
        env = np.sin(np.pi * t / duration) ** 0.9
        signal = signal * env

    else:  # neutral
        # Balanced F0, standard conversational speech prosody
        f0 = 150 + 15 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        signal = 0.65 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.10 * np.sin(3 * phase)
        env = np.sin(np.pi * t / duration)
        signal = signal * env

    # Add subtle white noise floor for realistic acoustic capture
    signal += 0.005 * rng.randn(n_samples)

    # Normalize amplitude
    max_val = np.max(np.abs(signal))
    if max_val > 1e-6:
        signal = signal / max_val * 0.95

    return signal.astype(np.float32)


def save_audio_wav(y: np.ndarray, file_path: Union[str, Path], sr: int = 22050) -> Path:
    """Save float32 audio waveform to standard 16-bit PCM WAV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path
