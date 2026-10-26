"""
config.py - Loads the API key and model names from the .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()      # optional: FREE, fast speech-to-text
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
WISPR_API_KEY = os.getenv("WISPR_API_KEY", "").strip()    # optional: Wispr Flow speech-to-text
WISPR_LANGUAGE = os.getenv("WISPR_LANGUAGE", "en").strip() # "en", "hi", ... or empty = auto-detect
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()  # image generation
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY", "").strip()  # video generation

# Models (change in .env if you want a different one)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")            # brain: text + images
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")    # your voice -> text
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")           # text -> AI voice
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")  # backup for images
RUNWAY_TEXT_MODEL = os.getenv("RUNWAY_TEXT_MODEL", "gen4.5")       # text -> video
RUNWAY_IMAGE_MODEL = os.getenv("RUNWAY_IMAGE_MODEL", "gen4_turbo") # image -> video
RUNWAY_DURATION = int(os.getenv("RUNWAY_DURATION", "5"))           # seconds

# Generated images and videos are saved here
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)
