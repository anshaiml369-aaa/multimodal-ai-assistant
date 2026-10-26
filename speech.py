"""
speech.py - Voice features:
  * transcribe()  your recorded voice  -> text   (Speech-to-Text)
                  uses Wispr Flow -> Groq -> OpenAI (first one with a key)
  * speak()       assistant's reply    -> audio  (Text-to-Speech, OpenAI)
"""

import base64
import io
import json
import re
import subprocess
import urllib.error
import urllib.request
import wave

from openai import OpenAI

import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

VOICES = ["coral", "nova", "alloy", "ash", "ballad", "echo", "fable", "onyx", "sage", "shimmer"]
WISPR_URL = "https://platform-api.wisprflow.ai/api/v1/dash/api"


# ------------------------------------------------------------------
# Speech-to-Text
# ------------------------------------------------------------------
def _to_wav_16k(audio_bytes: bytes) -> bytes:
    """Wispr Flow needs 16 kHz, mono, 16-bit WAV. Convert with ffmpeg if needed."""
    try:
        with wave.open(io.BytesIO(audio_bytes)) as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2):
                return audio_bytes            # already in the right format
    except (wave.Error, EOFError):
        pass                                  # not a WAV file (e.g. mp3) -> convert
    result = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0",
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", "-f", "wav", "pipe:1"],
        input=audio_bytes, capture_output=True, check=True,
    )
    return result.stdout


def _transcribe_wispr(audio_bytes: bytes) -> str:
    language = config.WISPR_LANGUAGE
    body = {
        "audio": base64.b64encode(_to_wav_16k(audio_bytes)).decode(),
        "context": {"app": {"type": "ai"}},
        # older API format - the server ignores it when "context" is present
        "properties": {"app_type": "ai", "language": language or "en"},
    }
    if language:
        body["language"] = [language]

    request = urllib.request.Request(
        WISPR_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {config.WISPR_API_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())["text"].strip()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="ignore")[:200]
        raise RuntimeError(f"Wispr Flow error {error.code}: {detail}") from error


def _transcribe_groq(audio_bytes: bytes, filename: str) -> str:
    # Groq's API is OpenAI-compatible: same SDK, just a different address and key
    groq = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    result = groq.audio.transcriptions.create(model=config.GROQ_STT_MODEL, file=(filename, audio_bytes))
    return result.text.strip()


def _transcribe_openai(audio_bytes: bytes, filename: str) -> str:
    result = client.audio.transcriptions.create(model=config.STT_MODEL, file=(filename, audio_bytes))
    return result.text.strip()


def _engines() -> list[tuple[str, callable]]:
    """Speech-to-text engines in order of preference (only those with a key)."""
    engines = []
    if config.WISPR_API_KEY:
        engines.append(("Wispr Flow", lambda audio, name: _transcribe_wispr(audio)))
    if config.GROQ_API_KEY:
        engines.append((f"Groq {config.GROQ_STT_MODEL}", _transcribe_groq))
    engines.append((config.STT_MODEL, _transcribe_openai))   # always available as backup
    return engines


def stt_engine() -> str:
    return _engines()[0][0]


def transcribe(audio_bytes: bytes, filename: str = "voice.wav") -> str:
    """Try each engine in turn; if one fails, fall back to the next."""
    engines = _engines()
    for name, engine in engines[:-1]:
        try:
            return engine(audio_bytes, filename)
        except Exception as error:
            print(f"[speech] {name} failed, trying next engine: {error}")
    return engines[-1][1](audio_bytes, filename)


# ------------------------------------------------------------------
# Text-to-Speech
# ------------------------------------------------------------------
def _clean_for_speech(text: str) -> str:
    """Remove markdown symbols (**, #, `) so the voice doesn't read them out."""
    text = re.sub(r"```.*?```", " (code shown on screen) ", text, flags=re.DOTALL)
    text = re.sub(r"[*#`_>|]", "", text)
    return text[:4000]   # API limit is 4096 characters


def speak(text: str, voice: str = "coral", style: str = "") -> bytes:
    response = client.audio.speech.create(
        model=config.TTS_MODEL,
        voice=voice,
        input=_clean_for_speech(text),
        instructions=style,          # e.g. "Speak like a kind teacher"
        response_format="mp3",
    )
    return response.content
