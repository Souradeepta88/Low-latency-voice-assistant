import os
import time
import json
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

st.set_page_config(page_title="Low-Latency Voice Assistant", page_icon="⚡")
st.title("⚡ Low-Latency Voice Assistant")
st.caption("Speak, and get a spoken reply back — every stage is timed so the latency claim is measured, not just asserted.")

GEMINI_MODEL = "gemini-2.5-flash"
RIME_MODEL = "mistv3"  # Rime's fastest, lowest-latency English model
RIME_TTS_URL = "https://users.rime.ai/v1/rime-tts"
RIME_VOICES_URL = "https://users.rime.ai/data/voices/voice_details.json"
SAMPLING_RATE = 22050  # lower than default 44100 -> smaller payload -> faster network transfer

CONVERSATIONS_FILE = "conversations.json"
BENCHMARK_HISTORY_FILE = "benchmark_history.json"
AUDIO_HISTORY_DIR = "audio_history"

os.makedirs(AUDIO_HISTORY_DIR, exist_ok=True)


def load_json_file(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json_file(path: str, data) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Could not save to {path}: {e}")


def create_new_conversation() -> str:
    """Create a new empty conversation, make it active, persist it, return its id."""
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "New conversation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "turns": [],
    }
    os.makedirs(os.path.join(AUDIO_HISTORY_DIR, conv_id), exist_ok=True)
    st.session_state.active_conversation_id = conv_id
    save_json_file(CONVERSATIONS_FILE, st.session_state.conversations)
    return conv_id


def delete_conversation(conv_id: str) -> None:
    """Delete one conversation and its saved audio files. Never touches others."""
    st.session_state.conversations.pop(conv_id, None)
    save_json_file(CONVERSATIONS_FILE, st.session_state.conversations)
    conv_audio_dir = os.path.join(AUDIO_HISTORY_DIR, conv_id)
    if os.path.exists(conv_audio_dir):
        for fname in os.listdir(conv_audio_dir):
            try:
                os.remove(os.path.join(conv_audio_dir, fname))
            except Exception:
                pass
        try:
            os.rmdir(conv_audio_dir)
        except Exception:
            pass
    if st.session_state.active_conversation_id == conv_id:
        remaining = list(st.session_state.conversations.keys())
        st.session_state.active_conversation_id = remaining[0] if remaining else None


# ---------------------------------------------------------------------------
# Load persisted conversations on startup
# ---------------------------------------------------------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = load_json_file(CONVERSATIONS_FILE, {})

if "active_conversation_id" not in st.session_state:
    existing_ids = list(st.session_state.conversations.keys())
    st.session_state.active_conversation_id = existing_ids[0] if existing_ids else None

if st.session_state.active_conversation_id is None:
    create_new_conversation()

# ---------------------------------------------------------------------------
# API keys — read only from .env, never typed into the app
# ---------------------------------------------------------------------------
load_dotenv()
gemini_key = os.environ.get("GEMINI_API_KEY", "")
rime_key = os.environ.get("RIME_API_KEY", "")

missing = []
if not gemini_key:
    missing.append("GEMINI_API_KEY")
if not rime_key:
    missing.append("RIME_API_KEY")

if missing:
    st.error(
        f"Missing: {', '.join(missing)}.\n\n"
        "Create a `.env` file next to this script with:\n\n"
        "```\nGEMINI_API_KEY=your-gemini-key\nRIME_API_KEY=your-rime-key\n```"
    )
    st.stop()

gemini_client = genai.Client(api_key=gemini_key)

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

    # List conversations, most recently created first
    sorted_convs = sorted(
        st.session_state.conversations.items(),
        key=lambda kv: kv[1].get("created_at", ""),
        reverse=True,
    )
    for conv_id, conv in sorted_convs:
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
                if not st.session_state.conversations:
                    create_new_conversation()
                st.rerun()

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None


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
# Voice turn — appends to the ACTIVE conversation only
# ---------------------------------------------------------------------------
active_conv = st.session_state.conversations[st.session_state.active_conversation_id]
st.subheader(f"🗨️ {active_conv.get('title', 'New conversation')}")

if "audio_input_counter" not in st.session_state:
    st.session_state.audio_input_counter = 0

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
            # so Gemini has context instead of treating each turn in isolation.
            history_contents = []
            for past_turn in active_conv["turns"]:
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

            # Save the actual reply audio to disk, scoped to this conversation
            conv_id = st.session_state.active_conversation_id
            turn_ts = datetime.now()
            audio_filename = None
            if audio_reply:
                audio_filename = f"{turn_ts.strftime('%Y%m%d_%H%M%S_%f')}.wav"
                conv_audio_dir = os.path.join(AUDIO_HISTORY_DIR, conv_id)
                os.makedirs(conv_audio_dir, exist_ok=True)
                audio_path = os.path.join(conv_audio_dir, audio_filename)
                try:
                    with open(audio_path, "wb") as f:
                        f.write(audio_reply)
                except Exception as e:
                    st.warning(f"Could not save audio to history: {e}")
                    audio_filename = None

            turn_record = {
                "timestamp": turn_ts.isoformat(timespec="seconds"),
                "transcript": parsed.get("transcript", ""),
                "reply": parsed["reply"],
                "audio_file": audio_filename,
                "stt_llm_ms": round(stt_llm_ms, 1),
                "tts_ms": round(tts_ms, 1),
                "total_ms": round(total_ms, 1),
            }

            conv = st.session_state.conversations[conv_id]
            conv["turns"].append(turn_record)
            # auto-title the conversation from its first turn, like a real chat app
            if conv["title"] == "New conversation" and turn_record["transcript"]:
                conv["title"] = turn_record["transcript"][:40] + ("…" if len(turn_record["transcript"]) > 40 else "")
            save_json_file(CONVERSATIONS_FILE, st.session_state.conversations)

            st.markdown(f"**You said:** {parsed.get('transcript', '')}")
            st.markdown(f"**Reply:** {parsed['reply']}")
            if audio_reply:
                st.audio(audio_reply, format="audio/wav")

            c1, c2, c3 = st.columns(3)
            c1.metric("STT + reasoning", f"{stt_llm_ms:.0f} ms")
            c2.metric("Rime TTS", f"{tts_ms:.0f} ms")
            c3.metric("Total (turn end → audio ready)", f"{total_ms:.0f} ms")

# ---------------------------------------------------------------------------
# This conversation's history — chat-style, with replayable audio
# ---------------------------------------------------------------------------
active_conv = st.session_state.conversations[st.session_state.active_conversation_id]  # re-fetch after possible update
if active_conv["turns"]:
    st.divider()
    st.subheader(f"💬 History ({len(active_conv['turns'])} turns in this conversation)")

    conv_audio_dir = os.path.join(AUDIO_HISTORY_DIR, st.session_state.active_conversation_id)
    for turn in active_conv["turns"]:
        with st.chat_message("user"):
            st.markdown(turn.get("transcript", ""))
        with st.chat_message("assistant"):
            st.markdown(turn.get("reply", ""))
            audio_filename = turn.get("audio_file")
            if audio_filename:
                audio_path = os.path.join(conv_audio_dir, audio_filename)
                if os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                else:
                    st.caption("⚠️ Saved audio file missing on disk.")
            st.caption(
                f"{turn.get('timestamp', '')} · STT+LLM {turn.get('stt_llm_ms', 0):.0f}ms · "
                f"TTS {turn.get('tts_ms', 0):.0f}ms · Total {turn.get('total_ms', 0):.0f}ms"
            )

# ---------------------------------------------------------------------------
# Repeatable TTS-only benchmark (isolates network + model latency from the
# rest of the pipeline; distinguishes cold vs warm runs, per Rime's own
# evidence guidance)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🧪 Repeatable TTS-only benchmark")
st.caption("Isolates Rime's latency alone (no mic, no Gemini) — a fixed phrase, run N times, cold run flagged separately.")

bench_text = st.text_input("Benchmark phrase", value="Your order will be ready in five minutes.")
num_runs = st.slider("Number of runs", min_value=1, max_value=10, value=5)

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "benchmark_history" not in st.session_state:
    st.session_state.benchmark_history = load_json_file(BENCHMARK_HISTORY_FILE, [])

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

    # Persist this batch to the running benchmark history file
    st.session_state.benchmark_history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phrase": bench_text,
        "runs": results,
    })
    save_json_file(BENCHMARK_HISTORY_FILE, st.session_state.benchmark_history)

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

if st.session_state.benchmark_history:
    with st.expander(f"📜 Benchmark history ({len(st.session_state.benchmark_history)} batches, saved to {BENCHMARK_HISTORY_FILE})"):
        for batch in reversed(st.session_state.benchmark_history):
            st.markdown(f"**{batch['timestamp']}** — \"{batch['phrase']}\"")
            st.table(batch["runs"])
            st.divider()
        if st.button("🗑️ Clear all benchmark history"):
            st.session_state.benchmark_history = []
            save_json_file(BENCHMARK_HISTORY_FILE, [])
            st.rerun()