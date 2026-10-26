# 🤖 Multimodal AI Assistant

A ChatGPT-style assistant that works with **text, voice, images and video** in one chat.
It can understand photos, answer out loud, and **create images and videos** when you ask.

## Features
- 💬 Chat with streaming replies (words appear as they're written)
- 🎤 Voice input: tap the mic in the chat box and just talk
- 🖼️ Image understanding: attach photos or use your 📷 webcam
- 🔊 Spoken replies with 10 voices
- 🎭 4 personalities: Friendly Assistant, Patient Teacher, Hinglish Buddy, Interview Coach
- 🎨 Image generation: *"Draw a cute astronaut cat"* (Gemini, backup OpenAI)
- 🎬 Video generation: *"Make a video of ocean waves"* or *"Animate that image"* (Runway)
- 🤖 The AI decides by itself when to create media (OpenAI tool / function calling)
- 🧠 Remembers the conversation (multi-turn memory)

## APIs used
| Part | Model | Job |
|------|-------|-----|
| Brain | `gpt-5.4-mini` | Understands text + images, writes the reply |
| Ears | Groq `whisper-large-v3-turbo` (backup: OpenAI) | Your voice → text |
| Mouth | `gpt-4o-mini-tts` | Reply → natural speech |
| Painter | Gemini `gemini-2.5-flash-image` (backup: OpenAI `gpt-image-1-mini`) | Text → image |
| Director | Runway `gen4.5` / `gen4_turbo` | Text → video, image → video |

Cost: a normal conversation costs well under ₹1 per message. Check platform.openai.com/pricing for current rates.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # paste your OpenAI key
streamlit run app.py
```

## Project structure
```
app.py       Streamlit chat UI (history, mic, images, webcam, audio player)
brain.py     Personalities + builds messages + streams the reply
speech.py    Speech-to-text and text-to-speech
media.py     Image generation (Gemini/OpenAI) and video generation (Runway)
config.py    Loads the key and model names from .env
```

## How it works
```
🎤 voice ─► transcribe() ─┐
💬 text ──────────────────┼─► brain.stream_reply() ─► reply text ─► speak() ─► 🔊
🖼️ image ─► base64 ───────┘        (with chat history + personality)
```

## Python concepts covered
- **Basic:** functions, lists, dictionaries, f-strings
- **Intermediate:** modules, `.env` secrets, regex, error handling, base64 encoding
- **Advanced:** generators + streaming, multimodal API messages, session state,
  speech-to-text / text-to-speech pipelines, callbacks and a web UI

## Notes
- The browser asks for **microphone/camera permission** the first time. Click Allow.
- If the voice doesn't autoplay, press ▶ on the audio player (some browsers block autoplay).
