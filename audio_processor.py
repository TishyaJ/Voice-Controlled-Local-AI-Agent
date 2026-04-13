"""
audio_processor.py — Handles all audio I/O and Speech-to-Text via Groq Whisper API.

Why Groq instead of local Whisper?
───────────────────────────────────
Running whisper-large locally on my CPU-only Windows machine requires ~4 GB RAM and
takes 30–120 s per utterance, making the UX unusable. Groq's hosted Whisper endpoint
delivers sub-second transcription with a free tier, so the CPU is kept free for
Ollama inference. This choice is documented in README.md.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz — Whisper's native sample rate
CHANNELS = 1                  # Mono
MAX_RECORDING_SECONDS = 30    # Safety cap on mic recordings
GROQ_MODEL = "whisper-large-v3"

# ── Groq Client (lazy init) ──────────────────────────────────────────────────
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Add it to your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


# ── Microphone Recording ─────────────────────────────────────────────────────

def record_audio(duration: int = 5) -> bytes:
    """
    Record *duration* seconds of audio from the default microphone.
    Returns raw WAV bytes (16-bit PCM, 16 kHz, mono).
    """
    duration = min(duration, MAX_RECORDING_SECONDS)
    frames = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()  # Block until done

    buf = io.BytesIO()
    sf.write(buf, frames, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ── Core Transcription ───────────────────────────────────────────────────────

def transcribe_audio(
    audio_bytes: bytes,
    file_ext: str = "wav",
    language: str = "en",
) -> Tuple[str, Optional[str]]:
    """
    Send *audio_bytes* to Groq's Whisper endpoint and return (transcript, error).

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file bytes (WAV, MP3, M4A, FLAC, OGG, WEBM).
    file_ext : str
        File extension without the dot, e.g. "wav", "mp3".
    language : str
        ISO-639-1 language code hint for Whisper (improves accuracy).

    Returns
    -------
    (transcript, None)   on success
    ("", error_message)  on any failure
    """
    if not audio_bytes:
        return "", "No audio data provided."

    ext = file_ext.lstrip(".").lower()
    # Groq accepts: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
    allowed_exts = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}
    if ext not in allowed_exts:
        ext = "wav"  # safe fallback

    try:
        client = _get_client()
        t0 = time.perf_counter()

        # Groq SDK expects a (filename, bytes, mimetype) tuple for file-like objects
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "mp4": "audio/mp4",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "webm": "audio/webm",
        }
        mime = mime_map.get(ext, "audio/wav")

        transcription = client.audio.transcriptions.create(
            file=(f"audio.{ext}", audio_bytes, mime),
            model=GROQ_MODEL,
            language=language,
            response_format="text",
        )

        elapsed = time.perf_counter() - t0
        transcript = str(transcription).strip()
        print(f"[STT] Groq transcription took {elapsed:.2f}s | text='{transcript[:80]}'")
        return transcript, None

    except EnvironmentError as env_err:
        return "", str(env_err)
    except Exception as exc:
        error_msg = f"Groq STT error: {type(exc).__name__}: {exc}"
        print(f"[STT] {error_msg}")
        return "", error_msg


# ── Convenience: transcribe a file path ──────────────────────────────────────

def transcribe_file(path: str | Path) -> Tuple[str, Optional[str]]:
    """Read *path* from disk and transcribe it."""
    p = Path(path)
    if not p.exists():
        return "", f"File not found: {p}"
    audio_bytes = p.read_bytes()
    return transcribe_audio(audio_bytes, file_ext=p.suffix)


# ── Convenience: transcribe an UploadedFile from Streamlit ──────────────────

def transcribe_uploaded(uploaded_file) -> Tuple[str, Optional[str]]:
    """
    Transcribe a ``streamlit.runtime.uploaded_file_manager.UploadedFile``.
    Works with any format Groq supports.
    """
    ext = Path(uploaded_file.name).suffix.lstrip(".")
    audio_bytes = uploaded_file.read()
    return transcribe_audio(audio_bytes, file_ext=ext)
