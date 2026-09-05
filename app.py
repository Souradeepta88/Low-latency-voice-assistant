import os
import time
import json
import uuid
import base64
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient

st.set_page_config(page_title="Low-Latency Voice Assistant", page_icon="⚡")
st.title("⚡ Low-Latency Voice Assistant")
st.caption("Speak, and get a spoken reply back — every stage is timed so the latency claim is measured, not just asserted.")

GEMINI_MODEL = "gemini-3.8-flash"
RIME_MODEL = "mistv3"  # Rime's fastest, lowest-latency English model
RIME_TTS_URL = "https://users.rime.ai/v1/rime-tts"
RIME_VOICES_URL = "https://users.rime.ai/data/voices/voice_details.json"
SAMPLING_RATE = 22050  # lower than default 44100 -> smaller payload -> faster network transfer
DB_NAME = "voice_assistant"

# ---------------------------------------------------------------------------
# API keys / connection string — read only from .env, never typed into the app
# ---------------------------------------------------------------------------
load_dotenv()
gemini_key = os.environ.get("GEMINI_API_KEY", "")
rime_key = os.environ.get("RIME_API_KEY", "")
mongodb_uri = os.environ.get("MONGODB_URI", "")

missing = []
if not gemini_key:
    missing.append("GEMINI_API_KEY")
if not rime_key:
    missing.append("RIME_API_KEY")
if not mongodb_uri:
    missing.append("MONGODB_URI")

if missing:
    st.error(
        f"Missing: {', '.join(missing)}.\n\n"
        "Create a `.env` file next to this script with:\n\n"
        "```\nGEMINI_API_KEY=your-gemini-key\nRIME_API_KEY=your-rime-key\n"
        "MONGODB_URI=your-mongodb-atlas-connection-string\n```"
    )
    st.stop()

gemini_client = genai.Client(api_key=gemini_key)


@st.cache_resource
def get_db():
    client = MongoClient(mongodb_uri)
    return client[DB_NAME]

db = get_db()
conversations_col = db["conversations"]
turns_col = db["turns"]
benchmark_col = db["benchmark_runs"]


# ---------------------------------------------------------------------------
# Conversation CRUD — all persisted to MongoDB, survives Streamlit Cloud
# sleep/wake cycles and redeploys, unlike local files
# ---------------------------------------------------------------------------
def create_new_conversation() -> str:
    conv_id = str(uuid.uuid4())
    conversations_col.insert_one({
        "_id": conv_id,
        "title": "New conversation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    st.session_state.active_conversation_id = conv_id
    return conv_id


def delete_conversation(conv_id: str) -> None:
    """Delete one conversation and its turns. Never touches other conversations."""
    conversations_col.delete_one({"_id": conv_id})
    turns_col.delete_many({"conversation_id": conv_id})
    if st.session_state.active_conversation_id == conv_id:
        remaining = list(conversations_col.find().sort("created_at", -1))
        st.session_state.active_conversation_id = remaining[0]["_id"] if remaining else None


def list_conversations():
    return list(conversations_col.find().sort("created_at", -1))


def get_turns(conv_id: str):
    return list(turns_col.find({"conversation_id": conv_id}).sort("timestamp", 1))


def add_turn(conv_id: str, transcript: str, reply: str, audio_bytes: bytes | None,
             stt_llm_ms: float, tts_ms: float, total_ms: float) -> None:
    turns_col.insert_one({
        "conversation_id": conv_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "transcript": transcript,
        "reply": reply,
        "audio_b64": base64.b64encode(audio_bytes).decode() if audio_bytes else None,
        "stt_llm_ms": round(stt_llm_ms, 1),
        "tts_ms": round(tts_ms, 1),
        "total_ms": round(total_ms, 1),
    })
    # Auto-title the conversation from its first turn, like a real chat app.
    # Only fires while the title is still the default, so it never overwrites
    # a title from a later turn.
    if transcript:
        title = transcript[:40] + ("…" if len(transcript) > 40 else "")
        conversations_col.update_one(
            {"_id": conv_id, "title": "New conversation"},
            {"$set": {"title": title}},
        )


# ---------------------------------------------------------------------------
# Active conversation bootstrap
# ---------------------------------------------------------------------------
if "active_conversation_id" not in st.session_state:
    existing = list_conversations()
    st.session_state.active_conversation_id = existing[0]["_id"] if existing else None

if st.session_state.active_conversation_id is None:
    create_new_conversation()

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if "audio_input_counter" not in st.session_state:
    st.session_state.audio_input_counter = 0

# ---------------------------------------------------------------------------
# Fetch Rime's LIVE voice catalog and pick one mistv3 English speaker at
# runtime, per Rime's own build rules (don't hardcode a stale speaker list).
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_mistv3_speaker(_api_key: str) -> str | None:
    try:
        resp = requests.get(
            RIME_VOICES_URL,
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        voices = resp.json()
        for v in voices:
            if v.get("modelId") == RIME_MODEL:
                return v.get("speaker")
    except Exception:
        pass
    return None

speaker = get_mistv3_speaker(rime_key)


def synthesize_speech(text: str) -> tuple[bytes | None, float]:
    """Call Rime mistv3 TTS. Returns (audio_bytes_or_None, elapsed_ms)."""
    if not speaker:
        return None, 0.0
    t_start = time.perf_counter()
    try:
        resp = requests.post(
            RIME_TTS_URL,
            headers={
                "Authorization": f"Bearer {rime_key}",
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
            json={
                "text": text,
                "modelId": RIME_MODEL,
                "speaker": speaker,
                "lang": "eng",
                "samplingRate": SAMPLING_RATE,
            },
            timeout=20,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return resp.content, elapsed_ms
    except Exception as e:
        st.error(f"Rime TTS error: {e}")
        return None, (time.perf_counter() - t_start) * 1000


# ---------------------------------------------------------------------------
# Sidebar: speech provider status + conversation switcher
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔊 Active speech provider")
    st.markdown(f"**Provider:** Rime\n\n**Model:** `{RIME_MODEL}` (fastest, lowest-latency)")
    if speaker:
        st.success(f"Voice: `{speaker}`")
    else:
        st.error("No live mistv3 voice found — check your Rime dashboard and hardcode a fallback speaker name.")

    st.divider()
    st.subheader("💬 Conversations")

    if st.button("➕ New conversation", use_container_width=True):
        create_new_conversation()
        st.rerun()

    st.divider()

    for conv in list_conversations():
        conv_id = conv["_id"]
        is_active = conv_id == st.session_state.active_conversation_id
        col_select, col_delete = st.columns([4, 1])
        with col_select:
            label = ("👉 " if is_active else "") + conv.get("title", "New conversation")
            if st.button(label, key=f"select_{conv_id}", use_container_width=True):
                st.session_state.active_conversation_id = conv_id
                st.rerun()
        with col_delete:
            if st.button("🗑️", key=f"delete_{conv_id}", help="Delete this conversation"):
                delete_conversation(conv_id)
                if not list_conversations():
                    create_new_conversation()
                st.rerun()

# ---------------------------------------------------------------------------
# Voice turn — appends to the ACTIVE conversation only
# ---------------------------------------------------------------------------
active_conv_id = st.session_state.active_conversation_id
active_conv = conversations_col.find_one({"_id": active_conv_id})
st.subheader(f"🗨️ {active_conv.get('title', 'New conversation')}")

if st.button("🎤 New message", use_container_width=True):
    st.session_state.audio_input_counter += 1
    st.session_state.last_audio_id = None
    st.rerun()

audio_value = st.audio_input("Speak your message", key=f"audio_input_{st.session_state.audio_input_counter}")

if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    audio_id = hash(audio_bytes)

    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id

        # t0: the moment we start processing = proxy for "end of user's turn"
        t0 = time.perf_counter()

        with st.spinner("Thinking..."):
            reply_prompt = (
                "You are a fast, helpful voice assistant for hands-busy, "
                "screen-light situations, having an ONGOING conversation with "
                "the user. Use the prior turns (given as conversation history) "
                "to remember context — names, numbers, preferences, or anything "
                "the user told you earlier in THIS conversation. Transcribe what "
                "the user just said, then reply directly to them, using that "
                "remembered context where relevant. CRITICAL: keep your reply to "
                "ONE short sentence, under 20 words — this is a real-time voice "
                "interface and long replies feel slow and are harder to listen "
                "to. Respond with ONLY a JSON object, no markdown fences: "
                '{"transcript": "...", "reply": "..."}'
            )

            # Build multi-turn history from prior turns in THIS conversation,
            # loaded from MongoDB, so Gemini has context instead of treating
            # each turn in isolation.
            prior_turns = get_turns(active_conv_id)
            history_contents = []
            for past_turn in prior_turns:
                if past_turn.get("transcript"):
                    history_contents.append(
                        types.Content(role="user", parts=[types.Part.from_text(text=past_turn["transcript"])])
                    )
                if past_turn.get("reply"):
                    history_contents.append(
                        types.Content(role="model", parts=[types.Part.from_text(text=past_turn["reply"])])
                    )

            current_turn_content = types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text="Respond to this, remembering our earlier conversation above."),
                ],
            )

            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=history_contents + [current_turn_content],
                    config=types.GenerateContentConfig(
                        system_instruction=reply_prompt,
                        temperature=0.4,
                        max_output_tokens=120,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                parsed = json.loads(response.text)
            except Exception as e:
                parsed = None
                st.error(f"Gemini error: {e}")

        t1 = time.perf_counter()  # after STT + reasoning
        stt_llm_ms = (t1 - t0) * 1000

        if parsed and parsed.get("reply"):
            with st.spinner("Speaking..."):
                audio_reply, tts_ms = synthesize_speech(parsed["reply"])

            t2 = time.perf_counter()  # after TTS bytes received
            total_ms = (t2 - t0) * 1000

            add_turn(
                active_conv_id,
                transcript=parsed.get("transcript", ""),
                reply=parsed["reply"],
                audio_bytes=audio_reply,
                stt_llm_ms=stt_llm_ms,
                tts_ms=tts_ms,
                total_ms=total_ms,
            )

            st.markdown(f"**You said:** {parsed.get('transcript', '')}")
            st.markdown(f"**Reply:** {parsed['reply']}")
            if audio_reply:
                st.audio(audio_reply, format="audio/wav")

            c1, c2, c3 = st.columns(3)
            c1.metric("STT + reasoning", f"{stt_llm_ms:.0f} ms")
            c2.metric("Rime TTS", f"{tts_ms:.0f} ms")
            c3.metric("Total (turn end → audio ready)", f"{total_ms:.0f} ms")

# ---------------------------------------------------------------------------
# This conversation's history — chat-style, with replayable audio, loaded
# fresh from MongoDB every run so it survives Streamlit Cloud restarts
# ---------------------------------------------------------------------------
turns = get_turns(active_conv_id)
if turns:
    st.divider()
    st.subheader(f"💬 History ({len(turns)} turns in this conversation)")

    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn.get("transcript", ""))
        with st.chat_message("assistant"):
            st.markdown(turn.get("reply", ""))
            if turn.get("audio_b64"):
                st.audio(base64.b64decode(turn["audio_b64"]), format="audio/wav")
            st.caption(
                f"{turn.get('timestamp', '')} · STT+LLM {turn.get('stt_llm_ms', 0):.0f}ms · "
                f"TTS {turn.get('tts_ms', 0):.0f}ms · Total {turn.get('total_ms', 0):.0f}ms"
            )

# ---------------------------------------------------------------------------
# Repeatable TTS-only benchmark (isolates network + model latency from the
# rest of the pipeline; distinguishes cold vs warm runs, per Rime's own
# evidence guidance). Also stored in MongoDB for persistence.
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🧪 Repeatable TTS-only benchmark")
st.caption("Isolates Rime's latency alone (no mic, no Gemini) — a fixed phrase, run N times, cold run flagged separately.")

bench_text = st.text_input("Benchmark phrase", value="Your order will be ready in five minutes.")
num_runs = st.slider("Number of runs", min_value=1, max_value=10, value=5)

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

col_run, col_reset = st.columns([3, 1])
with col_run:
    run_clicked = st.button("Run benchmark", use_container_width=True, type="primary")
with col_reset:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

if run_clicked:
    results = []
    progress = st.progress(0)
    for i in range(num_runs):
        _, elapsed = synthesize_speech(bench_text)
        results.append({"run": i + 1, "type": "cold" if i == 0 else "warm", "latency_ms": round(elapsed, 1)})
        progress.progress((i + 1) / num_runs)
    st.session_state.benchmark_results = results

    benchmark_col.insert_one({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phrase": bench_text,
        "runs": results,
    })

if reset_clicked:
    st.session_state.benchmark_results = None
    st.rerun()

if st.session_state.benchmark_results:
    results = st.session_state.benchmark_results
    st.table(results)
    warm_runs = [r["latency_ms"] for r in results if r["type"] == "warm"]
    if warm_runs:
        st.metric("Avg warm-run latency", f"{sum(warm_runs)/len(warm_runs):.0f} ms")
    st.metric("Cold-run latency (run 1)", f"{results[0]['latency_ms']:.0f} ms")

benchmark_batches = list(benchmark_col.find().sort("timestamp", -1))
if benchmark_batches:
    with st.expander(f"📜 Benchmark history ({len(benchmark_batches)} batches, stored in MongoDB)"):
        for batch in benchmark_batches:
            st.markdown(f"**{batch['timestamp']}** — \"{batch['phrase']}\"")
            st.table(batch["runs"])
            st.divider()
        if st.button("🗑️ Clear all benchmark history"):
            benchmark_col.delete_many({})
            st.rerun()