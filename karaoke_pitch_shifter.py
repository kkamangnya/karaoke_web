#!/usr/bin/env python3
"""
Karaoke pitch shifter with optional key analysis, vocal reduction, and denoising.

This script is designed for MP3/WAV karaoke tracks and keeps playback speed
unchanged while shifting pitch by semitone steps.

Core pipeline:
1. Load audio with soundfile when possible, fall back to pydub for MP3.
2. Optional mild vocal reduction for stereo material.
3. Optional spectral-gate style denoising.
4. Pitch shift with librosa's phase-vocoder-based implementation.
5. Save to WAV or MP3.

Usage examples:
  python karaoke_pitch_shifter.py -i song.mp3 -o song_plus2.wav -s 2
  python karaoke_pitch_shifter.py -i song.wav -o song_minus3.mp3 -s -3 --remove-vocals --denoise
  python karaoke_pitch_shifter.py -i song.mp3 -o out.wav --target-key Ebm
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover
    AudioSegment = None

try:
    import simpleaudio as sa
except Exception:  # pragma: no cover
    sa = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except Exception:  # pragma: no cover
    tk = None
    filedialog = messagebox = scrolledtext = ttk = None

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:  # pragma: no cover
    DND_FILES = None
    TkinterDnD = None


KEY_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)


@dataclass
class AudioData:
    y: np.ndarray  # shape: (n_samples,) or (n_channels, n_samples)
    sr: int


@dataclass
class ProcessedAudio:
    y: np.ndarray
    sr: int
    semitones: float
    detected_key: Optional[str]
    source_input: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Change karaoke pitch without changing playback speed."
    )
    parser.add_argument("-i", "--input", default=None, help="Input MP3/WAV file path.")
    parser.add_argument("-o", "--output", default=None, help="Output WAV/MP3 file path.")
    parser.add_argument(
        "-s",
        "--semitones",
        type=float,
        default=0.0,
        help="Pitch shift in semitones. Positive raises pitch, negative lowers it.",
    )
    parser.add_argument(
        "--target-key",
        default=None,
        help="Optional target key such as C, G#m, Ebm. Overrides --semitones when given.",
    )
    parser.add_argument(
        "--analyze-key",
        action="store_true",
        help="Print detected musical key before processing.",
    )
    parser.add_argument(
        "--remove-vocals",
        action="store_true",
        help="Apply a conservative center-cancel vocal reduction for stereo audio.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply a light spectral gate denoiser.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Optional resampling rate. If omitted, the input sample rate is preserved.",
    )
    parser.add_argument(
        "--output-format",
        choices=["wav", "mp3"],
        default=None,
        help="Force output format. If omitted, derived from output extension.",
    )
    parser.add_argument(
        "--mp3-bitrate",
        default="320k",
        help="MP3 export bitrate when output format is mp3.",
    )
    parser.add_argument(
        "--keep-stereo",
        action="store_true",
        help="Preserve stereo if possible. Default keeps channel count as loaded.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical user interface.",
    )
    return parser.parse_args()


def read_audio(path: str, target_sr: Optional[int] = None) -> AudioData:
    """
    Read audio using soundfile when possible.
    For MP3 or unsupported formats, fall back to pydub.
    """
    try:
        y, sr = sf.read(path, always_2d=True, dtype="float32")
        y = y.T  # shape: (channels, samples)
    except Exception:
        if AudioSegment is None:
            raise RuntimeError(
                "soundfile could not read the input and pydub is unavailable. "
                "Install ffmpeg and pydub for MP3 support."
            )
        seg = AudioSegment.from_file(path)
        sr = seg.frame_rate
        channels = seg.channels
        samples = np.array(seg.get_array_of_samples())
        if channels > 1:
            samples = samples.reshape((-1, channels)).T
        else:
            samples = samples[np.newaxis, :]
        sample_width = seg.sample_width
        max_val = float(1 << (8 * sample_width - 1))
        y = samples.astype(np.float32) / max_val
    if target_sr is not None and target_sr != sr:
        y = np.stack([librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in y], axis=0)
        sr = target_sr
    if y.shape[0] == 1:
        return AudioData(y=y[0], sr=sr)
    return AudioData(y=y, sr=sr)


def to_channel_first(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y[np.newaxis, :]
    if y.shape[0] <= y.shape[1]:
        return y
    return y.T


def from_channel_first(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.shape[0] == 1:
        return y[0]
    return y


def apply_vocal_reduction(y: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """
    Very conservative stereo vocal reduction:
    reduce the mid component while preserving side information.
    """
    y = to_channel_first(y)
    if y.shape[0] < 2:
        return from_channel_first(y)

    left = y[0]
    right = y[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid *= (1.0 - strength)
    left_new = mid + side
    right_new = mid - side
    out = np.stack([left_new, right_new], axis=0)
    return from_channel_first(out)


def estimate_noise_profile(magnitude: np.ndarray) -> np.ndarray:
    """
    Estimate a per-frequency noise floor from the lower-energy portion of frames.
    """
    # Use a low percentile per bin so that transient content is not treated as noise.
    return np.percentile(magnitude, 20, axis=1)


def spectral_gate(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Simple spectral gating denoiser.
    It is intentionally conservative to avoid musical artifacts.
    """
    y = to_channel_first(y)
    out_channels = []
    window = np.hanning(n_fft).astype(np.float32)
    for ch in y:
        stft = librosa.stft(ch, n_fft=n_fft, hop_length=hop_length, window=window, center=True)
        mag, phase = np.abs(stft), np.exp(1j * np.angle(stft))
        noise = estimate_noise_profile(mag)
        noise = noise[:, np.newaxis]
        threshold = noise * 1.8
        gain = np.clip((mag - threshold) / np.maximum(mag, 1e-8), 0.0, 1.0)
        # Light temporal smoothing to reduce musical noise.
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        gain = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=gain)
        gain = np.clip(gain, 0.0, 1.0)
        denoised = gain * mag * phase
        ch_out = librosa.istft(denoised, hop_length=hop_length, window=window, length=len(ch))
        out_channels.append(ch_out.astype(np.float32))
    out = np.stack(out_channels, axis=0)
    return from_channel_first(out)


def detect_key(y: np.ndarray, sr: int) -> Tuple[str, float]:
    """
    Detect likely key using a chroma template match against major/minor profiles.
    Returns (key_name, confidence_score).
    """
    y = np.mean(to_channel_first(y), axis=0)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_norm = chroma / np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8)
    profile_major = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
    profile_minor = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)
    scores = []
    for root in range(12):
        major = np.roll(profile_major, root)
        minor = np.roll(profile_minor, root)
        major_score = float(np.mean(chroma_norm.T @ major))
        minor_score = float(np.mean(chroma_norm.T @ minor))
        scores.append((major_score, root, "major"))
        scores.append((minor_score, root, "minor"))
    best = max(scores, key=lambda x: x[0])
    score, root, mode = best
    name = KEY_NAMES_FLAT[root] if "b" in KEY_NAMES_FLAT[root] else KEY_NAMES_SHARP[root]
    suffix = "" if mode == "major" else "m"
    return f"{name}{suffix}", score


def key_to_semitones(source_key: str, target_key: str) -> float:
    """
    Map a detected or assumed key to a target key and return the pitch shift in semitones.
    """
    def normalize(k: str) -> Tuple[int, str]:
        k = k.strip()
        mode = "major"
        if k.endswith("m") and not k.endswith("majm"):
            mode = "minor"
            k = k[:-1]
        token = k.replace("♯", "#").replace("♭", "b")
        aliases = {
            "Db": 1, "C#": 1,
            "Eb": 3, "D#": 3,
            "Gb": 6, "F#": 6,
            "Ab": 8, "G#": 8,
            "Bb": 10, "A#": 10,
        }
        if token in aliases:
            root = aliases[token]
        else:
            upper = token.upper()
            lookup = {name.upper(): i for i, name in enumerate(KEY_NAMES_SHARP)}
            if upper not in lookup:
                raise ValueError(f"Unsupported key name: {k}")
            root = lookup[upper]
        return root, mode

    src_root, src_mode = normalize(source_key)
    dst_root, dst_mode = normalize(target_key)
    if src_mode != dst_mode:
        # Keep relative major/minor matching simple for karaoke use.
        # If modes differ, only align pitch class.
        _ = (src_mode, dst_mode)
    return float(dst_root - src_root)


def shift_pitch(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Pitch shift without changing speed.
    librosa.effects.pitch_shift uses a phase-vocoder-based time-stretch + resampling pipeline.
    """
    y = to_channel_first(y)
    input_len = y.shape[1]
    shifted = []
    for ch in y:
        try:
            out = librosa.effects.pitch_shift(
                ch,
                sr=sr,
                n_steps=semitones,
                bins_per_octave=12,
                res_type="soxr_hq",
            )
        except Exception:
            out = librosa.effects.pitch_shift(
                ch,
                sr=sr,
                n_steps=semitones,
                bins_per_octave=12,
                res_type="kaiser_best",
            )
        shifted.append(out.astype(np.float32))
    shifted = [ch[:input_len] if len(ch) >= input_len else np.pad(ch, (0, input_len - len(ch))) for ch in shifted]
    out = np.stack(shifted, axis=0)
    return from_channel_first(out)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-9:
        return y
    return (0.98 * y / peak).astype(np.float32)


def save_audio(path: str, y: np.ndarray, sr: int, fmt: Optional[str] = None, bitrate: str = "320k") -> None:
    path_obj = Path(path)
    ext = (fmt or path_obj.suffix.lower().lstrip(".") or "wav").lower()
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)

    if ext == "wav":
        sf.write(path, y.T if y.ndim == 2 else y, sr, subtype="PCM_16")
        return

    if ext == "mp3":
        if AudioSegment is None:
            raise RuntimeError("pydub is required for MP3 export. Install pydub and ffmpeg.")
        pcm = (y.T if y.ndim == 2 else y).reshape(-1)
        channels = y.shape[0] if y.ndim == 2 else 1
        int16 = (pcm * 32767.0).astype(np.int16)
        seg = AudioSegment(
            int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=channels,
        )
        seg.export(path, format="mp3", bitrate=bitrate)
        return

    raise ValueError(f"Unsupported output format: {ext}")


def prepare_audio(args: argparse.Namespace, logger=print, progress=None) -> ProcessedAudio:
    def set_progress(value: int, message: Optional[str] = None) -> None:
        if progress is not None:
            progress(value, message)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    set_progress(5, "Loading audio")
    audio = read_audio(str(input_path), target_sr=args.sr)
    y = audio.y
    sr = audio.sr
    set_progress(15, "Audio loaded")

    if args.analyze_key or args.target_key:
        set_progress(22, "Analyzing key")
        detected_key, confidence = detect_key(y, sr)
        logger(f"Detected key: {detected_key}  (confidence: {confidence:.4f})")
        set_progress(30, "Key analyzed")
    else:
        detected_key = None

    if not args.keep_stereo:
        y = np.mean(to_channel_first(y), axis=0)
        logger("Downmixed to mono")
        set_progress(35, "Downmixed to mono")

    semitones = float(args.semitones)
    if args.target_key:
        source_key = detected_key or "C"
        semitones = key_to_semitones(source_key, args.target_key)
        logger(f"Target key: {args.target_key} -> pitch shift: {semitones:+.2f} semitones")

    if args.remove_vocals:
        set_progress(45, "Removing vocals")
        y = apply_vocal_reduction(y, strength=0.85)
        logger("Applied vocal reduction")
        set_progress(55, "Vocal reduction done")

    if args.denoise:
        set_progress(60, "Denoising")
        y = spectral_gate(y, sr)
        logger("Applied denoise")
        set_progress(72, "Denoise done")

    if abs(semitones) > 1e-9:
        set_progress(78, "Pitch shifting")
        y = shift_pitch(y, sr, semitones)
        logger(f"Applied pitch shift: {semitones:+.2f} semitones")
        set_progress(92, "Pitch shift done")

    set_progress(95, "Normalizing")
    y = normalize_audio(y)

    return ProcessedAudio(
        y=y,
        sr=sr,
        semitones=semitones,
        detected_key=detected_key,
        source_input=str(input_path),
    )


def process(args: argparse.Namespace, logger=print, progress=None) -> Path:
    prepared = prepare_audio(args, logger=logger, progress=progress)
    output_path = Path(args.output)
    if not output_path:
        raise ValueError("Output path is required.")

    output_fmt = args.output_format
    if output_fmt is None:
        output_fmt = output_path.suffix.lower().lstrip(".") or "wav"
    if progress is not None:
        progress(98, "Saving file")
    save_audio(str(output_path), prepared.y, prepared.sr, fmt=output_fmt, bitrate=args.mp3_bitrate)
    logger(f"Saved: {output_path}")
    if progress is not None:
        progress(100, "Done")
    return output_path


def build_default_output_path(input_path: str, output_format: str) -> str:
    src = Path(input_path)
    stem = src.stem
    parent = src.parent
    suffix = ".wav" if output_format == "wav" else ".mp3"
    return str(parent / f"{stem}_shifted{suffix}")


class KaraokeGUI:
    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        self.root.title("Karaoke Pitch Shifter")
        self.root.geometry("860x700")
        self.root.minsize(760, 620)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.semitones_var = tk.DoubleVar(value=0.0)
        self.target_key_var = tk.StringVar()
        self.output_format_var = tk.StringVar(value="wav")
        self.mp3_bitrate_var = tk.StringVar(value="320k")
        self.sr_var = tk.StringVar(value="")
        self.analyze_key_var = tk.BooleanVar(value=True)
        self.remove_vocals_var = tk.BooleanVar(value=False)
        self.denoise_var = tk.BooleanVar(value=False)
        self.keep_stereo_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.IntVar(value=0)
        self._preview_play_obj = None
        self._preview_temp_path: Optional[Path] = None
        self._preview_args_signature = None
        self._preview_ready_audio: Optional[ProcessedAudio] = None
        self._preview_session = 0
        self._state_lock = threading.Lock()

        self._build_ui()
        self._setup_drag_and_drop()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(5, weight=1)

        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(1, weight=1)

        title = ttk.Label(outer, text="Karaoke Pitch Shifter", font=("Segoe UI", 18, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(outer, text="Input file").grid(row=1, column=0, sticky="w")
        ttk.Entry(outer, textvariable=self.input_var).grid(row=1, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(outer, text="Browse", command=self.choose_input).grid(row=1, column=2, sticky="ew")

        ttk.Label(outer, text="Output file").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(outer, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(outer, text="Save As", command=self.choose_output).grid(row=2, column=2, sticky="ew", pady=(8, 0))

        controls = ttk.LabelFrame(outer, text="Processing", padding=12)
        controls.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Semitones").grid(row=0, column=0, sticky="w")
        semitone_scale = ttk.Scale(controls, from_=-12.0, to=12.0, variable=self.semitones_var, orient="horizontal")
        semitone_scale.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 8))
        ttk.Spinbox(controls, from_=-12.0, to=12.0, increment=0.5, textvariable=self.semitones_var, width=8).grid(row=0, column=4, sticky="e")

        ttk.Label(controls, text="Target key").grid(row=1, column=0, sticky="w", pady=(10, 0))
        key_combo = ttk.Combobox(
            controls,
            textvariable=self.target_key_var,
            values=["", "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B", "Cm", "C#m", "Dm", "D#m", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Abm", "Am", "A#m", "Bbm", "Bm"],
            width=10,
        )
        key_combo.grid(row=1, column=1, sticky="w", padx=(8, 8), pady=(10, 0))
        ttk.Checkbutton(controls, text="Analyze key", variable=self.analyze_key_var).grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(controls, text="Remove vocals", variable=self.remove_vocals_var).grid(row=1, column=3, sticky="w", pady=(10, 0))
        ttk.Checkbutton(controls, text="Denoise", variable=self.denoise_var).grid(row=1, column=4, sticky="w", pady=(10, 0))

        ttk.Label(controls, text="Output format").grid(row=2, column=0, sticky="w", pady=(10, 0))
        fmt_combo = ttk.Combobox(controls, textvariable=self.output_format_var, values=["wav", "mp3"], width=8, state="readonly")
        fmt_combo.grid(row=2, column=1, sticky="w", padx=(8, 8), pady=(10, 0))
        ttk.Label(controls, text="MP3 bitrate").grid(row=2, column=2, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.mp3_bitrate_var, width=10).grid(row=2, column=3, sticky="w", pady=(10, 0))
        ttk.Checkbutton(controls, text="Keep stereo", variable=self.keep_stereo_var).grid(row=2, column=4, sticky="w", pady=(10, 0))

        ttk.Label(controls, text="Sample rate").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.sr_var, width=10).grid(row=3, column=1, sticky="w", padx=(8, 8), pady=(10, 0))
        ttk.Label(controls, text="Leave blank to preserve original").grid(row=3, column=2, columnspan=3, sticky="w", pady=(10, 0))

        actions = ttk.Frame(outer)
        actions.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 8))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="기본 경로 생성", command=self.fill_default_output).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="미리듣기", command=self.preview_processing).grid(row=0, column=1, sticky="e", padx=(0, 8))
        ttk.Button(actions, text="정지", command=self.stop_preview).grid(row=0, column=2, sticky="e", padx=(0, 8))
        ttk.Button(actions, text="저장", command=self.save_current).grid(row=0, column=3, sticky="e")

        drop_frame = ttk.LabelFrame(outer, text="Drop Zone", padding=12)
        drop_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        drop_frame.columnconfigure(0, weight=1)
        self.drop_label = ttk.Label(
            drop_frame,
            text="Drag and drop an MP3/WAV file here",
            anchor="center",
            padding=12,
        )
        self.drop_label.grid(row=0, column=0, sticky="ew")

        progress_frame = ttk.Frame(outer)
        progress_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        progress_frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.status_var, width=28, anchor="e")
        self.progress_label.grid(row=0, column=1, padx=(8, 0), sticky="e")

        log_frame = ttk.LabelFrame(outer, text="Log", padding=8)
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=18, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        status = ttk.Label(outer, textvariable=self.status_var, relief="sunken", anchor="w")
        status.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self.input_var.trace_add("write", lambda *_: self._auto_update_output())
        self.output_format_var.trace_add("write", lambda *_: self._auto_update_output())

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.status_var.set(message)

    def set_progress(self, value: int, message: Optional[str] = None) -> None:
        self.progress_var.set(max(0, min(100, int(value))))
        if message:
            self.status_var.set(message)

    def _current_signature(self) -> tuple:
        args = self._collect_args()
        return (
            args.input,
            args.semitones,
            args.target_key,
            args.analyze_key,
            args.remove_vocals,
            args.denoise,
            args.sr,
            args.output_format,
            args.mp3_bitrate,
            args.keep_stereo,
        )

    def _is_session_active(self, session_id: int) -> bool:
        with self._state_lock:
            return session_id == self._preview_session

    def _ensure_output_path(self, args: argparse.Namespace) -> None:
        if not args.output:
            self.fill_default_output()
            args.output = self.output_var.get().strip()

    def _reset_preview(self) -> None:
        with self._state_lock:
            self._preview_session += 1
        self._preview_ready_audio = None
        self._preview_args_signature = None
        self._cleanup_preview_temp()

    def _cleanup_preview_temp(self) -> None:
        if self._preview_play_obj is not None:
            try:
                self._preview_play_obj.stop()
            except Exception:
                pass
            self._preview_play_obj = None
        if self._preview_temp_path is not None:
            try:
                self._preview_temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._preview_temp_path = None

    def stop_preview(self) -> None:
        with self._state_lock:
            self._preview_session += 1
        self._cleanup_preview_temp()
        self.status_var.set("Preview stopped")
        self.log("Preview stopped")

    def _play_preview_file(self, temp_path: Path) -> None:
        if sa is None:
            raise RuntimeError(
                "Preview playback requires the simpleaudio package. "
                "Install it with `python -m pip install simpleaudio`."
            )
        self._cleanup_preview_temp()
        wave_obj = sa.WaveObject.from_wave_file(str(temp_path))
        self._preview_play_obj = wave_obj.play()
        self._preview_temp_path = temp_path

    def _prepare_in_background(self, args: argparse.Namespace, on_ready, on_error, session_id: int) -> None:
        def worker() -> None:
            try:
                self.root.after(0, self.set_progress, 0, "Processing...")
                prepared = prepare_audio(
                    args,
                    logger=lambda msg: self.root.after(0, self.log, msg),
                    progress=lambda value, message=None: self.root.after(0, self.set_progress, value, message),
                )
                def deliver_ready() -> None:
                    if not self._is_session_active(session_id):
                        return
                    on_ready(prepared)

                self.root.after(0, deliver_ready)
            except Exception as exc:
                def deliver_error() -> None:
                    if not self._is_session_active(session_id):
                        return
                    on_error(exc)

                self.root.after(0, deliver_error)

        threading.Thread(target=worker, daemon=True).start()

    def choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose input audio",
            filetypes=[("Audio files", "*.wav *.mp3"), ("WAV", "*.wav"), ("MP3", "*.mp3"), ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)
            self._reset_preview()

    def choose_output(self) -> None:
        fmt = self.output_format_var.get().strip().lower() or "wav"
        ext = ".mp3" if fmt == "mp3" else ".wav"
        path = filedialog.asksaveasfilename(
            title="Choose output file",
            defaultextension=ext,
            filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def fill_default_output(self) -> None:
        inp = self.input_var.get().strip()
        if not inp:
            return
        self.output_var.set(build_default_output_path(inp, self.output_format_var.get().strip().lower() or "wav"))

    def _auto_update_output(self) -> None:
        if self.output_var.get().strip():
            return
        inp = self.input_var.get().strip()
        if not inp:
            return
        self.output_var.set(build_default_output_path(inp, self.output_format_var.get().strip().lower() or "wav"))

    def _collect_args(self) -> argparse.Namespace:
        sr_text = self.sr_var.get().strip()
        sr = int(sr_text) if sr_text else None
        output_format = self.output_format_var.get().strip().lower() or None
        return argparse.Namespace(
            input=self.input_var.get().strip() or None,
            output=self.output_var.get().strip() or None,
            semitones=float(self.semitones_var.get()),
            target_key=self.target_key_var.get().strip() or None,
            analyze_key=bool(self.analyze_key_var.get()),
            remove_vocals=bool(self.remove_vocals_var.get()),
            denoise=bool(self.denoise_var.get()),
            sr=sr,
            output_format=output_format,
            mp3_bitrate=self.mp3_bitrate_var.get().strip() or "320k",
            keep_stereo=bool(self.keep_stereo_var.get()),
            gui=True,
        )

    def _normalize_dropped_path(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        return raw

    def _handle_dropped_files(self, data: str) -> None:
        if not data:
            return
        try:
            paths = self.root.tk.splitlist(data)
        except Exception:
            paths = data.split()
        first = paths[0] if paths else ""
        path = self._normalize_dropped_path(first)
        if path:
            self.input_var.set(path)
            if not self.output_var.get().strip():
                self._auto_update_output()
            self._reset_preview()
            self.log(f"Dropped: {path}")

    def _setup_drag_and_drop(self) -> None:
        if TkinterDnD is None or DND_FILES is None:
            self.drop_label.configure(text="Drag and drop unavailable. Install tkinterdnd2 for this feature.")
            return
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", lambda event: self._handle_dropped_files(event.data))
            self.drop_label.configure(text="Drop audio anywhere in this window")
        except Exception:
            self.drop_label.configure(text="Drag and drop initialization failed. Browse button still works.")

    def preview_processing(self) -> None:
        args = self._collect_args()
        if not args.input:
            messagebox.showerror("Missing input", "Please choose an input audio file.")
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.status_var.set("Preparing preview...")
        self.progress_var.set(0)
        with self._state_lock:
            self._preview_session += 1
            session_id = self._preview_session
        requested_signature = self._current_signature()
        self._preview_args_signature = requested_signature

        def on_ready(prepared: ProcessedAudio) -> None:
            try:
                if requested_signature != self._current_signature():
                    self.log("Preview ignored because settings changed")
                    return
                fd, temp_name = tempfile.mkstemp(suffix=".wav", prefix="karaoke_preview_")
                os.close(fd)
                temp = Path(temp_name)
                save_audio(str(temp), prepared.y, prepared.sr, fmt="wav")
                self._preview_ready_audio = prepared
                self._play_preview_file(temp)
                self._preview_args_signature = self._current_signature()
                self.status_var.set("Preview playing")
                self.progress_var.set(100)
                self.log("Preview ready and playing")
            except Exception as exc:
                self.status_var.set("Error")
                messagebox.showerror("Preview failed", str(exc))
                self.log(f"Error: {exc}")

        def on_error(exc: Exception) -> None:
            self.status_var.set("Error")
            messagebox.showerror("Preview failed", str(exc))
            self.log(f"Error: {exc}")

        self.log("Starting preview preparation")
        self._prepare_in_background(args, on_ready, on_error, session_id=session_id)

    def save_current(self) -> None:
        args = self._collect_args()
        if not args.input:
            messagebox.showerror("Missing input", "Please choose an input audio file.")
            return
        self._ensure_output_path(args)
        if not args.output:
            messagebox.showerror("Missing output", "Please choose an output file.")
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.status_var.set("Preparing to save...")
        self.progress_var.set(0)

        current_signature = self._current_signature()
        use_cached = self._preview_ready_audio is not None and self._preview_args_signature == current_signature

        def finish_save(prepared: ProcessedAudio) -> None:
            try:
                output_path = Path(args.output)
                output_fmt = args.output_format or output_path.suffix.lower().lstrip(".") or "wav"
                self.set_progress(98, "Saving file")
                save_audio(str(output_path), prepared.y, prepared.sr, fmt=output_fmt, bitrate=args.mp3_bitrate)
                self.set_progress(100, "Done")
                self.status_var.set("Done")
                self.log(f"Saved: {output_path}")
            except Exception as exc:
                self.status_var.set("Error")
                messagebox.showerror("Save failed", str(exc))
                self.log(f"Error: {exc}")

        def on_error(exc: Exception) -> None:
            self.status_var.set("Error")
            messagebox.showerror("Save failed", str(exc))
            self.log(f"Error: {exc}")

        if use_cached:
            self.log("Using previewed audio for save")
            finish_save(self._preview_ready_audio)
            return

        self.log("Preview not available for current settings. Reprocessing before save.")
        with self._state_lock:
            self._preview_session += 1
            session_id = self._preview_session
        self._prepare_in_background(
            args,
            on_ready=finish_save,
            on_error=on_error,
            session_id=session_id,
        )


def launch_gui() -> None:
    if tk is None:
        raise RuntimeError("tkinter is not available in this Python installation.")
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    KaraokeGUI(root)
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.gui or not args.input:
        launch_gui()
        return
    if not args.output:
        args.output = build_default_output_path(args.input, args.output_format or "wav")
    process(args)


if __name__ == "__main__":
    main()
