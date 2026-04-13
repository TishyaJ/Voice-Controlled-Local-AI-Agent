"""
app.py — Voice-Controlled Local AI Agent
========================================
Split-screen Streamlit UI:
  Left  (60%) : Audio input (mic / upload), pipeline logs, approve/deny controls
  Right (40%) : Chat & action history (persistent via StreamlitChatMessageHistory)

Architecture
────────────
1. Audio Input  → audio_processor.py  → Groq Whisper → transcript
2. Transcript   → LangChain prompt    → Ollama qwen   → JSON array of actions
3. JSON actions → tools.py dispatcher → tool results
4. Results      → left-column logs  +  right-column chat history

Bonus features implemented:
  ✅ Compound Commands  — LLM returns a JSON array; we loop over every action
  ✅ Human-in-the-Loop  — "create_file" / "write_code" require Approve / Deny
  ✅ Graceful Degradation — all exceptions caught; falls back to general_chat
  ✅ Memory              — StreamlitChatMessageHistory + st.session_state
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import OllamaLLM

from audio_processor import transcribe_audio, transcribe_uploaded
from tools import (
    OUTPUT_DIR,
    create_file,
    generate_and_write_code,
    general_chat,
    summarize_text,
    write_code,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
SAMPLE_RATE = 16_000
CHANNELS = 1

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Header strip ── */
    .agent-header {
        background: linear-gradient(90deg, #7928ca, #ff0080);
        border-radius: 16px;
        padding: 1.2rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 8px 32px rgba(121,40,202,0.4);
    }
    .agent-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
        letter-spacing: -0.5px;
    }
    .agent-header p {
        margin: 0;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.75);
    }

    /* ── Glass cards ── */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    .glass-card h3 {
        margin: 0 0 0.75rem 0;
        font-size: 0.9rem;
        font-weight: 600;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Log pipeline badges ── */
    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .pipeline-step:last-child { border-bottom: none; }
    .step-badge {
        min-width: 90px;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        text-align: center;
        white-space: nowrap;
    }
    .badge-transcript { background: rgba(56,189,248,0.2); color: #38bdf8; border: 1px solid #38bdf8; }
    .badge-intent     { background: rgba(167,139,250,0.2); color: #a78bfa; border: 1px solid #a78bfa; }
    .badge-action     { background: rgba(52,211,153,0.2);  color: #34d399; border: 1px solid #34d399; }
    .badge-result     { background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-error      { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid #f87171; }
    .step-content { color: rgba(255,255,255,0.85); font-size: 0.88rem; line-height: 1.5; }

    /* ── Chat bubbles ── */
    .chat-msg { display: flex; gap: 0.75rem; margin-bottom: 1rem; }
    .chat-msg.human { flex-direction: row-reverse; }
    .avatar {
        width: 34px; height: 34px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
    }
    .avatar-human { background: linear-gradient(135deg,#7928ca,#ff0080); }
    .avatar-ai    { background: linear-gradient(135deg,#0ea5e9,#6366f1); }
    .bubble {
        max-width: 80%;
        padding: 0.75rem 1rem;
        border-radius: 14px;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .bubble-human {
        background: linear-gradient(135deg,#7928ca,#ff0080);
        color: #fff;
        border-bottom-right-radius: 4px;
    }
    .bubble-ai {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.15);
        color: rgba(255,255,255,0.9);
        border-bottom-left-radius: 4px;
    }

    /* ── Approve / Deny buttons ── */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }

    /* ── Spinner overlay text ── */
    .recording-badge {
        display: inline-flex; align-items: center; gap: 0.5rem;
        background: rgba(239,68,68,0.2); border: 1px solid #ef4444;
        color: #ef4444; border-radius: 20px;
        padding: 4px 14px; font-size: 0.78rem; font-weight: 600;
        animation: pulse 1.2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

    /* ── Section divider ── */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* ── Scrollable chat pane ── */
    .chat-scroll {
        max-height: 520px;
        overflow-y: auto;
        padding-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization ──────────────────────────────────────────────

def _init_state():
    defaults = {
        "pipeline_logs": [],       # list of dicts for left-column logs
        "pending_actions": [],     # Human-in-the-Loop queue
        "recording": False,
        "recorded_bytes": None,
        "history_obj": None,       # StreamlitChatMessageHistory (set below)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# StreamlitChatMessageHistory for LangChain memory
if st.session_state.history_obj is None:
    st.session_state.history_obj = StreamlitChatMessageHistory(key="langchain_messages")

msg_history: StreamlitChatMessageHistory = st.session_state.history_obj

# Ensure the key exists even if history_obj was reused
if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []


# ── LangChain / Ollama helpers ────────────────────────────────────────────────

def _get_llm() -> OllamaLLM:
    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_predict=256,   # JSON responses are short — cap output tokens
        num_ctx=2048,      # Smaller context window = faster CPU inference
    )


SYSTEM_PROMPT = """\
You are an intent-classification and action-planning AI. Given a user's voice transcript, \
analyze it and return a JSON **array** of actions to perform.

Each element in the array must be one of:

  {{ "intent": "create_file",           "filename": "<name>" }}
  {{ "intent": "write_code",            "description": "<what to code>", "filename": "<target.py>" }}
  {{ "intent": "summarize_text",        "text": "<text to summarize>",   "save_to": "<optional filename or null>" }}
  {{ "intent": "general_chat",          "message": "<user message>" }}

Rules:
- ALWAYS output a valid JSON array, even for a single action.
- For compound commands ("summarize X and save to Y"), output multiple elements.
- If the intent is unclear, default to general_chat.
- Do NOT include any explanation outside the JSON array.
- Use snake_case intent names exactly as shown above.

Conversation context (last turns):
{history}

Current user transcript:
"{transcript}"

JSON array of actions:"""


def build_prompt(transcript: str) -> str:
    messages = st.session_state.get("langchain_messages", [])
    history_lines = []
    for m in messages[-6:]:  # last 6 messages for context
        if isinstance(m, HumanMessage):
            history_lines.append(f"Human: {m.content}")
        elif isinstance(m, AIMessage):
            history_lines.append(f"AI: {m.content}")
    history_str = "\n".join(history_lines) or "(none)"
    return SYSTEM_PROMPT.format(history=history_str, transcript=transcript)


def parse_llm_response(raw: str) -> list[dict]:
    """
    Extract a JSON array from *raw* LLM output.
    Falls back to a single general_chat action on any parse failure.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip("` \n")

    # Try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Try extracting first [...] block
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Fallback
    return [{"intent": "general_chat", "message": raw[:300]}]


HUMAN_IN_LOOP_INTENTS = {"create_file", "write_code"}


def classify_and_plan(transcript: str) -> list[dict]:
    """Send transcript to Ollama and parse the resulting action plan."""
    llm = _get_llm()
    prompt = build_prompt(transcript)
    try:
        raw = llm.invoke(prompt)
        actions = parse_llm_response(str(raw))
        # Validate each action has at least an "intent" key
        valid = []
        for a in actions:
            if isinstance(a, dict) and "intent" in a:
                valid.append(a)
            else:
                valid.append({"intent": "general_chat", "message": str(a)})
        return valid or [{"intent": "general_chat", "message": transcript}]
    except Exception as exc:
        return [{"intent": "general_chat", "message": f"(LLM error: {exc}) — {transcript}"}]


# ── Tool Execution ────────────────────────────────────────────────────────────

def execute_action(action: dict) -> str:
    """Dispatch *action* to the appropriate tool function."""
    intent = action.get("intent", "general_chat")
    history = [
        {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in st.session_state.get("langchain_messages", [])
    ]

    if intent == "create_file":
        return create_file(action.get("filename", "untitled.txt"))

    elif intent == "write_code":
        desc = action.get("description", "")
        fname = action.get("filename", "generated_code.py")
        return generate_and_write_code(desc, fname)

    elif intent == "summarize_text":
        text = action.get("text", "")
        save_to = action.get("save_to") or None
        return summarize_text(text, save_to=save_to)

    elif intent == "general_chat":
        msg = action.get("message", "")
        return general_chat(msg, history=history)

    else:
        # Unknown intent → graceful degradation
        return general_chat(
            f"I received an unknown command type '{intent}'. Let me try to help anyway: "
            + action.get("message", ""),
            history=history,
        )


# ── Log helpers ───────────────────────────────────────────────────────────────

def add_log(badge_class: str, label: str, content: str):
    st.session_state.pipeline_logs.append(
        {"badge": badge_class, "label": label, "content": content}
    )


def clear_logs():
    st.session_state.pipeline_logs = []


# ── Mic Recording (runs in background thread) ─────────────────────────────────

def _do_record(duration: int):
    """Record audio in a background thread and store bytes in session_state."""
    frames = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    import io
    buf = io.BytesIO()
    sf.write(buf, frames, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    st.session_state.recorded_bytes = buf.read()
    st.session_state.recording = False


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(audio_bytes: bytes, file_ext: str = "wav"):
    """Transcribe → classify → execute (with Human-in-the-Loop gating)."""
    clear_logs()

    # 1. Transcribe
    with st.spinner("🎤 Transcribing audio via Groq Whisper…"):
        transcript, stt_error = transcribe_audio(audio_bytes, file_ext=file_ext)

    if stt_error or not transcript.strip():
        error_msg = stt_error or "Empty transcription returned — please try again."
        add_log("badge-error", "STT Error", error_msg)
        # Graceful degradation: still show something useful
        msg_history.add_user_message("(unintelligible audio)")
        msg_history.add_ai_message(f"⚠️ {error_msg}")
        return

    add_log("badge-transcript", "Transcript", transcript)
    msg_history.add_user_message(transcript)

    # 2. Classify & plan
    with st.spinner("🧠 Classifying intent with Ollama…"):
        actions = classify_and_plan(transcript)

    intents_str = ", ".join(a.get("intent", "?") for a in actions)
    add_log("badge-intent", "Intent(s)", intents_str)

    # 3. Separate safe vs. gated actions
    safe_actions = []
    gated_actions = []
    for action in actions:
        if action.get("intent") in HUMAN_IN_LOOP_INTENTS:
            gated_actions.append(action)
        else:
            safe_actions.append(action)

    # 4. Execute safe actions immediately
    all_results: list[str] = []
    for action in safe_actions:
        intent = action.get("intent", "?")
        add_log("badge-action", "Action", f"Executing: {intent}")
        result = execute_action(action)
        add_log("badge-result", "Result", result[:400])
        all_results.append(result)

    # 5. Queue gated actions for Human-in-the-Loop
    if gated_actions:
        st.session_state.pending_actions.extend(gated_actions)
        pending_str = ", ".join(
            f"`{a.get('intent')} → {a.get('filename', '')}`" for a in gated_actions
        )
        add_log(
            "badge-intent",
            "⏸ Awaiting Approval",
            f"File operations pending your approval: {pending_str}",
        )

    # 6. Save AI response to history (safe actions only for now)
    if all_results:
        combined = "\n\n---\n\n".join(all_results)
        msg_history.add_ai_message(combined)


# ══════════════════════════════════════════════════════════════════════════════
# UI Layout
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="agent-header">
        <div style="font-size:2.2rem">🎙️</div>
        <div>
            <h1>Voice AI Agent</h1>
            <p>Speak a command → intent classification → local tool execution</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([3, 2])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Controls + Pipeline Logs
# ══════════════════════════════════════════════════════════════════════════════
with left_col:

    # ── Mic Input ─────────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card"><h3>🎤 Microphone Input</h3>', unsafe_allow_html=True)

    rec_duration = st.slider(
        "Recording duration (seconds)", min_value=3, max_value=30, value=5, step=1,
        key="rec_duration",
    )

    col_rec, col_status = st.columns([1, 2])
    with col_rec:
        record_btn = st.button(
            "⏺ Record", key="btn_record", use_container_width=True,
            type="primary",
        )
    with col_status:
        if st.session_state.recording:
            st.markdown(
                '<span class="recording-badge">🔴 Recording…</span>',
                unsafe_allow_html=True,
            )

    if record_btn and not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.recorded_bytes = None
        frames = sd.rec(
            int(rec_duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        sd.wait()
        import io as _io
        buf = _io.BytesIO()
        sf.write(buf, frames, SAMPLE_RATE, format="WAV", subtype="PCM_16")

        buf.seek(0)
        st.session_state.recorded_bytes = buf.read()
        st.session_state.recording = False
        st.rerun()

    # Process recorded audio
    if st.session_state.recorded_bytes and not st.session_state.recording:
        st.audio(st.session_state.recorded_bytes, format="audio/wav")
        if st.button("▶ Process Recording", key="btn_process_rec", use_container_width=True):
            run_pipeline(st.session_state.recorded_bytes, file_ext="wav")
            st.session_state.recorded_bytes = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── File Upload ───────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card"><h3>📁 Upload Audio File</h3>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload WAV / MP3 / M4A / OGG",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        key="audio_upload",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        st.audio(uploaded)
        if st.button("▶ Transcribe & Process", key="btn_process_upload", use_container_width=True, type="primary"):
            uploaded.seek(0)
            audio_bytes = uploaded.read()
            ext = Path(uploaded.name).suffix.lstrip(".")
            run_pipeline(audio_bytes, file_ext=ext)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Pipeline Log ──────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card"><h3>🔍 Pipeline Log</h3>', unsafe_allow_html=True)

    if st.session_state.pipeline_logs:
        log_html = ""
        for entry in st.session_state.pipeline_logs:
            content_escaped = (
                entry["content"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            log_html += f"""
            <div class="pipeline-step">
                <span class="step-badge {entry['badge']}">{entry['label']}</span>
                <span class="step-content">{content_escaped}</span>
            </div>"""
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<p style='color:rgba(255,255,255,0.35);font-size:0.85rem;'>"
            "Logs will appear here after processing audio…</p>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Human-in-the-Loop: Approve / Deny ─────────────────────────────────────
    if st.session_state.pending_actions:
        st.markdown(
            '<div class="glass-card"><h3>⚠️ Actions Awaiting Approval</h3>',
            unsafe_allow_html=True,
        )
        st.warning(
            "The following file operations require your approval before execution:",
            icon="🔐",
        )

        to_remove: list[int] = []

        for idx, action in enumerate(st.session_state.pending_actions):
            intent = action.get("intent", "?")
            fname = action.get("filename", "?")
            desc = action.get("description", "")

            with st.container():
                st.markdown(
                    f"**Action {idx + 1}:** `{intent}` → `{fname}`"
                    + (f"\n\n> {desc}" if desc else ""),
                )
                approve_col, deny_col, _ = st.columns([1, 1, 2])
                with approve_col:
                    if st.button(
                        "✅ Approve",
                        key=f"approve_{idx}",
                        use_container_width=True,
                        type="primary",
                    ):
                        result = execute_action(action)
                        add_log("badge-action", "Approved", f"{intent} → {fname}")
                        add_log("badge-result", "Result", result[:400])
                        msg_history.add_ai_message(
                            f"✅ Approved and executed: `{intent}` on `{fname}`\n\n{result}"
                        )
                        to_remove.append(idx)
                        st.rerun()
                with deny_col:
                    if st.button(
                        "❌ Deny",
                        key=f"deny_{idx}",
                        use_container_width=True,
                    ):
                        add_log("badge-error", "Denied", f"User denied: {intent} → {fname}")
                        msg_history.add_ai_message(
                            f"❌ User denied the action: `{intent}` on `{fname}`"
                        )
                        to_remove.append(idx)
                        st.rerun()
            st.divider()

        # Remove processed actions (reverse order to keep indices valid)
        for i in sorted(to_remove, reverse=True):
            st.session_state.pending_actions.pop(i)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Output Folder Status ──────────────────────────────────────────────────
    with st.expander("📂 Output folder contents", expanded=False):
        files = list(OUTPUT_DIR.rglob("*"))
        if files:
            for f in sorted(files):
                if f.is_file():
                    rel = f.relative_to(OUTPUT_DIR)
                    size = f.stat().st_size
                    st.markdown(f"- `{rel}` ({size:,} bytes)")
        else:
            st.markdown("*(empty — files created by the agent appear here)*")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Chat & Action History
# ══════════════════════════════════════════════════════════════════════════════
with right_col:
    st.markdown(
        '<div class="glass-card" style="height:100%;min-height:600px;">'
        "<h3>💬 Chat &amp; Action History</h3>",
        unsafe_allow_html=True,
    )

    messages = st.session_state.get("langchain_messages", [])

    if not messages:
        st.markdown(
            "<p style='color:rgba(255,255,255,0.35);font-size:0.85rem;margin-top:1rem;'>"
            "Your conversation history will appear here.<br><br>"
            "Try recording a command like:<br>"
            "• <em>\"Summarize this text: …\"</em><br>"
            "• <em>\"Create a Python file called utils.py\"</em><br>"
            "• <em>\"Write a retry function and save it to retry.py\"</em>"
            "</p>",
            unsafe_allow_html=True,
        )
    else:
        chat_html = '<div class="chat-scroll">'
        for msg in messages:
            if isinstance(msg, HumanMessage):
                chat_html += f"""
                <div class="chat-msg human">
                    <div class="avatar avatar-human">🧑</div>
                    <div class="bubble bubble-human">{msg.content}</div>
                </div>"""
            elif isinstance(msg, AIMessage):
                # Render simple markdown-style bold/code
                content = msg.content.replace("**", "<strong>", 1)
                content = content.replace("**", "</strong>", 1)
                content = content.replace("`", "<code>").replace("`", "</code>")
                content = content.replace("\n", "<br>")
                chat_html += f"""
                <div class="chat-msg">
                    <div class="avatar avatar-ai">🤖</div>
                    <div class="bubble bubble-ai">{content}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # ── Text chat fallback ────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='font-size:0.78rem;color:rgba(255,255,255,0.45);margin-bottom:0.5rem;'>"
        "Type a message directly (no audio needed):</p>",
        unsafe_allow_html=True,
    )

    with st.form(key="text_chat_form", clear_on_submit=True):
        text_input = st.text_input(
            "Message",
            placeholder="Ask anything or give a command…",
            label_visibility="collapsed",
            key="text_chat_input",
        )
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and text_input.strip():
        msg_history.add_user_message(text_input)
        actions = classify_and_plan(text_input)
        add_log("badge-transcript", "Text Input", text_input)
        add_log("badge-intent", "Intent(s)", ", ".join(a.get("intent", "?") for a in actions))

        all_results: list[str] = []
        gated: list[dict] = []
        for action in actions:
            if action.get("intent") in HUMAN_IN_LOOP_INTENTS:
                gated.append(action)
            else:
                result = execute_action(action)
                all_results.append(result)
                add_log("badge-result", "Result", result[:400])

        if gated:
            st.session_state.pending_actions.extend(gated)

        combined = "\n\n---\n\n".join(all_results)
        if combined:
            msg_history.add_ai_message(combined)

        st.rerun()

    # ── Reset conversation ────────────────────────────────────────────────────
    if st.button("🗑 Clear History", key="btn_clear_history"):
        st.session_state["langchain_messages"] = []
        clear_logs()
        st.session_state.pending_actions = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
