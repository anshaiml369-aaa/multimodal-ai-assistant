"""
app.py - Multimodal AI Assistant (talk, type, show images; it answers by voice).

Run with:   streamlit run app.py
"""

import streamlit as st

import config
from brain import PERSONALITIES, stream_reply
from media import MediaError, generate_image, generate_video, image_engines, video_engines
from speech import VOICES, speak, stt_engine, transcribe

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.brand {font-size: 2.1rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #8B7CFF, #3EC6FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.sub {text-align: center; color: #9A9AA5; margin-top: -0.5rem;}
</style>
<div class="brand">🤖 Multimodal AI Assistant</div>
<div class="sub">Text 💬 · Voice 🎤 · Images 🖼️ · Video 🎬 — all in one chat</div>
""", unsafe_allow_html=True)

if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it to the .env file and restart.")
    st.stop()

# Chat history lives in session_state so it survives Streamlit reruns
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    personality = st.selectbox("Personality", list(PERSONALITIES))
    voice = st.selectbox("Voice", VOICES)
    speak_replies = st.toggle("🔊 Speak replies out loud", value=True)

    st.divider()
    with st.expander("📷 Show something with your webcam"):
        snapshot = st.camera_input("Take a photo", label_visibility="collapsed")
        st.caption("The photo is sent along with your next message.")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.caption(f"Brain: `{config.CHAT_MODEL}`  \nEars: `{stt_engine()}`  \nVoice: `{config.TTS_MODEL}`  \n"
               f"Images: `{' → '.join(image_engines()) or '❌ no key'}`  \n"
               f"Video: `{' → '.join(video_engines()) or '❌ no key'}`")


def show_message(msg: dict, autoplay: bool = False) -> None:
    is_user = msg["role"] == "user"
    with st.chat_message(msg["role"], avatar="🧑" if is_user else "🤖"):
        if msg.get("voice_input"):
            st.caption("🎤 (spoken)")
        if msg["text"]:
            st.markdown(msg["text"])
        for data, _ in msg.get("images", []):
            st.image(data, width=260 if is_user else None)
        for video_path in msg.get("videos", []):
            st.video(video_path)
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3", autoplay=autoplay)


def last_image() -> bytes | None:
    """The most recent picture in the chat (uploaded or generated) - used for 'animate it'."""
    for msg in reversed(st.session_state.history):
        if msg.get("images"):
            return msg["images"][-1][0]
    return None


def run_action(action: dict, reply_msg: dict) -> None:
    """Carry out a tool request from the AI and show the result."""
    prompt = action["args"].get("prompt", "")
    if action["name"] == "generate_image":
        with st.spinner("🎨 Creating your image..."):
            data, engine = generate_image(prompt)
        st.image(data)
        st.caption(f"🎨 Made with {engine}")
        reply_msg["images"].append((data, "image/png"))
        reply_msg["notes"].append(f"I generated an image of: {prompt}")

    elif action["name"] == "generate_video":
        source = last_image() if action["args"].get("animate_last_image") else None
        label = "Animating the image" if source else "Creating your video"
        with st.spinner(f"🎬 {label}... (1-3 minutes)"):
            path, engine = generate_video(prompt, source)
        st.video(str(path))
        st.caption(f"🎬 Made with {engine}")
        reply_msg["videos"].append(str(path))
        reply_msg["notes"].append(f"I generated a video of: {prompt}")


# ---------------- Show old messages ----------------
if not st.session_state.history:
    st.info("👋 Chat, ask about a photo, or ask me to create things!  \n"
            "Try: *“Draw a cute astronaut cat”* · *“Make a video of ocean waves at sunset”* · "
            "*“Now animate that image”*  \n🎤 **Speak to type** puts your words in the box — then press **Send**.")
for msg in st.session_state.history:
    show_message(msg)
chat_area = st.container()   # new messages appear here (above the message box)

# ---------------- Message box: type OR speak-to-type ----------------
st.session_state.setdefault("draft", "")
st.session_state.setdefault("mic_id", 0)     # changing an id gives a fresh, empty widget
st.session_state.setdefault("files_id", 0)


def on_voice_typing():
    """Recording finished -> speech to text -> add the text to the message box."""
    recording = st.session_state.get(f"mic_{st.session_state.mic_id}")
    if not recording:
        return
    try:
        spoken = transcribe(recording.getvalue())
    except Exception as error:
        st.session_state.voice_error = f"Could not convert speech: {error}"
        return
    if spoken:
        st.session_state.draft = f"{st.session_state.draft} {spoken}".strip()
        st.session_state.draft_spoken = True
    else:
        st.session_state.voice_error = "I couldn't hear anything - check your microphone and try again."
    st.session_state.mic_id += 1             # reset the recorder for the next sentence


def on_send():
    """Send button -> move the message into 'pending' and clear the box."""
    files = st.session_state.get(f"files_{st.session_state.files_id}") or []
    images = [(f.getvalue(), f.type) for f in files]
    # Webcam photo: attach it once (remember its id so it isn't sent twice)
    if snapshot and st.session_state.get("used_snapshot") != snapshot.file_id:
        images.append((snapshot.getvalue(), "image/jpeg"))
        st.session_state.used_snapshot = snapshot.file_id

    st.session_state.pending = {
        "role": "user",
        "text": st.session_state.draft.strip(),
        "images": images,
        "voice_input": st.session_state.pop("draft_spoken", False),
    }
    st.session_state.draft = ""
    st.session_state.files_id += 1


st.divider()
st.text_area("✍️ Your message", key="draft", height=100,
             placeholder="Type here... or click 🎤 below and speak - your words will be typed here")
if "voice_error" in st.session_state:
    st.warning(st.session_state.pop("voice_error"))

col_mic, col_files = st.columns(2)
with col_mic:
    st.audio_input("🎤 Speak to type", key=f"mic_{st.session_state.mic_id}",
                   on_change=on_voice_typing,
                   help="Click to start, click again to stop. Your words appear in the box above.")
with col_files:
    st.file_uploader("🖼️ Attach images", type=["png", "jpg", "jpeg", "webp"],
                     accept_multiple_files=True, key=f"files_{st.session_state.files_id}")

st.button("📤 Send", type="primary", use_container_width=True, on_click=on_send)

# ---------------- Answer the pending message ----------------
user_msg = st.session_state.pop("pending", None)
if user_msg and (user_msg["text"] or user_msg["images"]):
    with chat_area:
        st.session_state.history.append(user_msg)
        show_message(user_msg)

        with st.chat_message("assistant", avatar="🤖"):
            try:
                actions = []   # filled if the AI decides to make an image/video
                reply = st.write_stream(stream_reply(st.session_state.history, personality, actions))
                reply = reply if isinstance(reply, str) else ""
                reply_msg = {"role": "assistant", "text": reply, "audio": None,
                             "images": [], "videos": [], "notes": []}

                if speak_replies and reply:
                    with st.spinner("🔊 Speaking..."):
                        reply_msg["audio"] = speak(reply, voice, style=PERSONALITIES[personality][1])
                    st.audio(reply_msg["audio"], format="audio/mp3", autoplay=True)

                for action in actions:
                    try:
                        run_action(action, reply_msg)
                    except MediaError as error:
                        st.error(f"❌ {error}")
                        reply_msg["notes"].append(f"Creating the {action['name']} failed: {error}")

                st.session_state.history.append(reply_msg)
            except Exception as error:
                st.error(f"❌ Something went wrong: {error}")
                st.session_state.history.pop()   # remove the unanswered message
