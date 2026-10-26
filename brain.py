"""
brain.py - The "brain" of the assistant: sends text + images to OpenAI
and streams the reply back word by word (like ChatGPT typing).
It can also decide to call TOOLS to create an image or a video.
"""

import base64
import json

from openai import OpenAI

import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

# personality name -> (system prompt, how the voice should sound)
PERSONALITIES = {
    "😊 Friendly Assistant": (
        "You are a friendly, helpful AI assistant.",
        "Speak in a warm, friendly and cheerful tone.",
    ),
    "👩‍🏫 Patient Teacher": (
        "You are a patient teacher for engineering students. Explain step by step "
        "with simple examples. End with one quick question to check understanding.",
        "Speak slowly and clearly like a kind teacher.",
    ),
    "🇮🇳 Hinglish Buddy": (
        "You are a fun friend who replies in Hinglish (a natural mix of Hindi and "
        "English written in Roman script).",
        "Speak casually and energetically, like a friend from India.",
    ),
    "💼 Interview Coach": (
        "You are a strict but helpful technical interviewer for software jobs. "
        "Ask one question at a time, then give honest feedback on the answer.",
        "Speak in a calm, professional tone.",
    ),
}

RULES = (
    " Keep answers short and conversational (2-6 sentences) unless the user asks "
    "for detail. When the user shares an image, describe what is relevant to their question."
    " You can create pictures with the generate_image tool and videos with the "
    "generate_video tool. Use them ONLY when the user asks you to draw, create, generate "
    "or make an image/picture/photo or a video/animation. Write the tool prompt in "
    "detailed English. Before calling a tool, say one short friendly sentence."
)

# Tools the AI is allowed to call (OpenAI "function calling")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Create a new picture from a text description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string",
                               "description": "Detailed description: subject, style, lighting, colors."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": "Create a short video clip (about 5 seconds).",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string",
                               "description": "Detailed description of the motion, scene and camera."},
                    "animate_last_image": {
                        "type": "boolean",
                        "description": "True if the user wants to animate / bring to life the most "
                                       "recent image in the conversation (uploaded or generated).",
                    },
                },
                "required": ["prompt", "animate_last_image"],
            },
        },
    },
]


def image_to_part(image_bytes: bytes, mime_type: str) -> dict:
    """Images are sent to the API as base64 text inside a 'data URL'."""
    encoded = base64.b64encode(image_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def to_api_messages(history: list[dict], personality: str) -> list[dict]:
    """
    Converts our chat history into the format OpenAI expects.
    history item = {"role": "user"/"assistant", "text": str, "images": [(bytes, mime)]}
    """
    system_prompt = PERSONALITIES[personality][0] + RULES
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if msg["role"] == "user" and msg.get("images"):
            parts = [{"type": "text", "text": msg["text"] or "What is in this image?"}]
            parts += [image_to_part(data, mime) for data, mime in msg["images"]]
            messages.append({"role": "user", "content": parts})
        else:
            # Tell the AI about images/videos it made earlier (so "animate it" makes sense)
            text = msg["text"] + "".join(f"\n[{note}]" for note in msg.get("notes", []))
            messages.append({"role": msg["role"], "content": text})
    return messages


def stream_reply(history: list[dict], personality: str, actions: list):
    """
    Generator: yields the reply in small pieces as it is being written.
    If the AI decides to use a tool, the tool request is added to `actions`,
    e.g. {"name": "generate_image", "args": {"prompt": "..."}}.
    """
    stream = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=to_api_messages(history, personality),
        tools=TOOLS,
        stream=True,
    )
    tool_calls = {}   # tool calls also arrive in small pieces -> join them up
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
        for call in delta.tool_calls or []:
            piece = tool_calls.setdefault(call.index, {"name": "", "args": ""})
            piece["name"] += call.function.name or ""
            piece["args"] += call.function.arguments or ""

    for piece in tool_calls.values():
        actions.append({"name": piece["name"], "args": json.loads(piece["args"] or "{}")})
