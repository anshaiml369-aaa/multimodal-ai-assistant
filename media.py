"""
media.py - Creates new media for the assistant:
  * generate_image()  text -> picture   (Gemini, backup: OpenAI)
  * generate_video()  text or picture -> video   (Runway)
"""

import base64
import io
import urllib.request
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from PIL import Image

import config


class MediaError(Exception):
    """Raised when an image or video cannot be created."""


def _new_path(extension: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.MEDIA_DIR / f"{stamp}.{extension}"


# ------------------------------------------------------------------
# Images
# ------------------------------------------------------------------
def _image_gemini(prompt: str) -> bytes:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9"),
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
    raise MediaError("Gemini did not return an image (the prompt may have been blocked).")


def _image_openai(prompt: str) -> bytes:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.images.generate(
        model=config.OPENAI_IMAGE_MODEL, prompt=prompt, size="1536x1024", quality="low")
    return base64.b64decode(response.data[0].b64_json)


def image_engines() -> list[str]:
    engines = []
    if config.GEMINI_API_KEY:
        engines.append("Gemini")
    if config.OPENAI_API_KEY:
        engines.append("OpenAI")
    return engines


def generate_image(prompt: str) -> tuple[bytes, str]:
    """Returns (image bytes, name of the engine that made it). Tries Gemini first."""
    errors = []
    for name, make in [("Gemini", _image_gemini), ("OpenAI", _image_openai)]:
        if name not in image_engines():
            continue
        try:
            data = make(prompt)
            _new_path("png").write_bytes(data)      # keep a copy in media/
            return data, name
        except Exception as error:
            print(f"[media] {name} image failed: {error}")
            errors.append(f"{name}: {str(error)[:150]}")
    raise MediaError("Could not create the image. " + (" | ".join(errors) or "No image API key set."))


# ------------------------------------------------------------------
# Videos
# ------------------------------------------------------------------
def _image_to_data_uri(image_bytes: bytes) -> str:
    """Shrink the picture (max 1280px) and turn it into text that Runway accepts."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _video_runway(prompt: str, image_bytes: bytes | None) -> Path:
    if not config.RUNWAY_API_KEY:
        raise MediaError("RUNWAY_API_KEY is missing in the .env file.")
    from runwayml import RunwayML, TaskFailedError

    client = RunwayML(api_key=config.RUNWAY_API_KEY)
    try:
        if image_bytes:
            task = client.image_to_video.create(
                model=config.RUNWAY_IMAGE_MODEL,
                prompt_image=_image_to_data_uri(image_bytes),
                prompt_text=prompt[:1000],
                ratio="1280:720",
                duration=config.RUNWAY_DURATION,
            )
        else:
            task = client.text_to_video.create(
                model=config.RUNWAY_TEXT_MODEL,
                prompt_text=prompt[:1000],
                ratio="1280:720",
                duration=config.RUNWAY_DURATION,
            )
        result = task.wait_for_task_output()      # checks the job until the video is ready
    except TaskFailedError as error:
        raise MediaError(f"Runway could not make this video: {error.task_details.failure}") from error
    except Exception as error:
        if "enough credits" in str(error):
            raise MediaError("Your Runway account has no credits. Add credits at "
                             "dev.runwayml.com → Billing, then try again.") from error
        raise MediaError(f"Runway error: {error}") from error

    out_path = _new_path("mp4")
    urllib.request.urlretrieve(result.output[0], out_path)
    return out_path


def video_engines() -> list[str]:
    engines = []
    if config.RUNWAY_API_KEY:
        engines.append("Runway")
    # (OpenAI Sora was a backup here, but its API shut down on 24 Sep 2026)
    return engines


def generate_video(prompt: str, image_bytes: bytes | None = None) -> tuple[Path, str]:
    """
    Text -> video, or animate a picture if image_bytes is given. Takes 1-3 minutes.
    Returns (video file, name of the engine that made it).
    """
    errors = []
    for name, make in [("Runway", _video_runway)]:   # add more video engines here
        if name not in video_engines():
            continue
        try:
            return make(prompt, image_bytes), name
        except Exception as error:
            print(f"[media] {name} video failed: {error}")
            errors.append(f"{name}: {str(error)[:200]}")
    raise MediaError("Could not create the video. " + (" | ".join(errors) or "No video API key set."))
