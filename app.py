"""
app.py — Voice-Controlled Local AI Agent (Chat-First UI)
=========================================================
Layout:
  Sidebar     : Mic input, file upload, output folder, clear history
  Main canvas : st.chat_message bubbles + st.chat_input pinned at bottom
  Per-message : Collapsible ⚙️ expander shows pipeline steps

All LangChain / Ollama / tool logic is unchanged.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import OllamaLLM

from audio_processor import transcribe_audio
from tools import (
    OUTPUT_DIR,
    create_file,
    generate_and_write_code,
    general_chat,
    summarize_text,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
SAMPLE_RATE   = 16_000
CHANNELS      = 1
HUMAN_IN_LOOP = {"create_file", "write_code"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon=":microphone:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* dark gradient background */
.stApp {
    background: linear-gradient(140deg, #0f0c29 0%, #302b63 55%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] h1 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg,#a78bfa,#f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

/* ── Sidebar section labels ── */
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin: 1.2rem 0 0.4rem 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    margin-bottom: 0.6rem !important;
    padding: 0.8rem 1rem !important;
}

/* user bubble accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid #a78bfa !important;
}
/* assistant bubble accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid #34d399 !important;
}

/* ── Pipeline step pills ── */
.pill {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-bottom: 4px;
    white-space: nowrap;
}
.pill-transcript { background:rgba(56,189,248,.18); color:#38bdf8; border:1px solid #38bdf8; }
.pill-intent     { background:rgba(167,139,250,.18); color:#a78bfa; border:1px solid #a78bfa; }
.pill-action     { background:rgba(52,211,153,.18);  color:#34d399; border:1px solid #34d399; }
.pill-result     { background:rgba(251,191,36,.18);  color:#fbbf24; border:1px solid #fbbf24; }
.pill-error      { background:rgba(248,113,113,.18); color:#f87171; border:1px solid #f87171; }

.pipeline-row {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.82rem;
    color: rgba(255,255,255,0.8);
}
.pipeline-row:last-child { border-bottom: none; }

/* ── Pending action card ── */
.pending-card {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
}
.pending-card p { margin: 0 0 0.6rem 0; font-size: 0.88rem; color: rgba(255,255,255,0.85); }

/* ── Chat input bar ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
}

/* ── Recording badge ── */
.rec-badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(239,68,68,.15); border:1px solid #ef4444;
    color:#ef4444; border-radius:20px; padding:3px 12px;
    font-size:0.75rem; font-weight:600;
    animation: blink 1.1s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }

/* ── Output file pill ── */
.file-pill {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.4);
    color:#818cf8; border-radius:8px; padding:3px 10px;
    font-size:0.78rem; font-family:monospace;
    margin:2px 4px 2px 0;
}

/* ── Scrollable chat ── */
section.main > div { padding-bottom: 5rem; }

/* ── Header gradient bar ── */
.agent-topbar {
    background: linear-gradient(90deg,#7928ca,#ff0080);
    border-radius:12px; padding:0.9rem 1.4rem;
    margin-bottom:1rem;
    display:flex; align-items:center; gap:0.8rem;
    box-shadow:0 6px 24px rgba(121,40,202,.35);
}
.agent-topbar h2 { margin:0; font-size:1.3rem; font-weight:700; color:#fff; }
.agent-topbar p  { margin:0; font-size:0.78rem; color:rgba(255,255,255,0.72); }

/* expander header */
[data-testid="stExpander"] summary {
    font-size:0.8rem !important;
    color: rgba(255,255,255,0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        # Each item: {"role": "user"|"assistant", "content": str,
        #             "pipeline": [...], "is_pending": bool, "action": dict}
        "messages": [],
        "pending_actions": [],
        "recorded_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ══════════════════════════════════════════════════════════════════════════════
# LangChain / Ollama helpers  (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

def _get_llm() -> OllamaLLM:
    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_predict=256,
        num_ctx=2048,
    )


SYSTEM_PROMPT = """\
You are an intent-classification and action-planning AI. Given a user's voice transcript, \
analyze it and return a JSON **array** of actions to perform.

Each element in the array must be one of:

  {{ "intent": "create_file",        "filename": "<name>" }}
  {{ "intent": "write_code",         "description": "<what to code>", "filename": "<target.py>" }}
  {{ "intent": "summarize_text",     "text": "<text to summarize>", "save_to": "<filename or null>" }}
  {{ "intent": "general_chat",       "message": "<user message>" }}

Rules:
- ALWAYS output a valid JSON array, even for a single action.
- For compound commands output multiple elements.
- If unclear, default to general_chat.
- Output ONLY the JSON array — no markdown, no explanation.

Conversation context (last turns):
{history}

User transcript:
"{transcript}"

JSON array:"""


def _build_history_str() -> str:
    lines = []
    for m in st.session_state.messages[-8:]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:200]}")
    return "\n".join(lines) or "(none)"


def _parse_llm(raw: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip("` \n")
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return [{"intent": "general_chat", "message": raw[:300]}]


def classify(transcript: str) -> list[dict]:
    llm = _get_llm()
    prompt = SYSTEM_PROMPT.format(
        history=_build_history_str(), transcript=transcript
    )
    try:
        raw  = llm.invoke(prompt)
        acts = _parse_llm(str(raw))
        valid = [a for a in acts if isinstance(a, dict) and "intent" in a]
        return valid or [{"intent": "general_chat", "message": transcript}]
    except Exception as exc:
        return [{"intent": "general_chat", "message": f"(LLM error: {exc}) {transcript}"}]


def execute_action(action: dict) -> str:
    intent  = action.get("intent", "general_chat")
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    if intent == "create_file":
        return create_file(action.get("filename", "untitled.txt"))
    if intent == "write_code":
        return generate_and_write_code(
            action.get("description", ""), action.get("filename", "code.py")
        )
    if intent == "summarize_text":
        return summarize_text(action.get("text", ""), save_to=action.get("save_to") or None)
    if intent == "general_chat":
        return general_chat(action.get("message", ""), history=history)
    return general_chat(
        f"Unknown intent '{intent}', trying to answer anyway: " + action.get("message", ""),
        history=history,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner  (returns pipeline log entries + result text)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(audio_bytes: bytes, file_ext: str = "wav"):
    """Full STT → classify → execute pipeline. Mutates st.session_state."""

    pipeline: list[dict] = []

    # ── 1. STT ───────────────────────────────────────────────────────────────
    with st.spinner("Transcribing via Groq Whisper..."):
        transcript, err = transcribe_audio(audio_bytes, file_ext=file_ext)

    if err or not transcript.strip():
        msg = err or "Empty transcription — please try again."
        pipeline.append({"badge": "pill-error", "label": "STT Error", "content": msg})
        _add_message("user", "(unintelligible audio)", [])
        _add_message("assistant", f"[ERROR] {msg}", pipeline)
        return

    pipeline.append({"badge": "pill-transcript", "label": "Transcript", "content": transcript})
    _add_message("user", transcript, [])

    # ── 2. Classify ──────────────────────────────────────────────────────────
    with st.spinner("Classifying intent..."):
        actions = classify(transcript)

    pipeline.append({
        "badge": "pill-intent",
        "label": "Intent(s)",
        "content": ", ".join(a.get("intent", "?") for a in actions),
    })

    # ── 3. Route actions ─────────────────────────────────────────────────────
    safe_results: list[str] = []
    for action in actions:
        intent = action.get("intent", "?")
        if intent in HUMAN_IN_LOOP:
            st.session_state.pending_actions.append(action)
            pipeline.append({
                "badge": "pill-intent",
                "label": "[PENDING]",
                "content": f"{intent} -> {action.get('filename','?')} (awaiting approval)",
            })
        else:
            pipeline.append({"badge": "pill-action", "label": "Action", "content": f"Executing: {intent}"})
            result = execute_action(action)
            pipeline.append({"badge": "pill-result", "label": "Result", "content": result[:400]})
            safe_results.append(result)

    combined = "\n\n---\n\n".join(safe_results) if safe_results else ""
    if not combined and st.session_state.pending_actions:
        combined = "File operation is waiting for your **Approve / Deny**."

    _add_message("assistant", combined, pipeline)


def _add_message(role: str, content: str, pipeline: list[dict]):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "pipeline": pipeline,
    })


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Toolbox
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# Voice AI Agent")
    st.caption("Speak · Classify · Execute")
    st.divider()

    # ── Mic ──────────────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">Microphone</p>', unsafe_allow_html=True)

    rec_duration = st.slider("Duration (s)", 3, 30, 5, key="rec_dur", label_visibility="collapsed")

    col_btn, col_status = st.columns([1, 1])
    with col_btn:
        rec_btn = st.button("Record", key="btn_rec", use_container_width=True, type="primary")
    with col_status:
        if st.session_state.get("_recording"):
            st.markdown('<span class="rec-badge">REC</span>', unsafe_allow_html=True)

    if rec_btn:
        st.session_state["_recording"] = True
        with st.spinner(f"Recording {rec_duration}s…"):
            frames = sd.rec(
                int(rec_duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
            )
            sd.wait()
        buf = io.BytesIO()
        sf.write(buf, frames, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        st.session_state.recorded_bytes = buf.read()
        st.session_state["_recording"] = False
        st.rerun()

    if st.session_state.recorded_bytes:
        st.audio(st.session_state.recorded_bytes, format="audio/wav")
        if st.button("Process Recording", key="btn_proc_rec", use_container_width=True):
            data = st.session_state.recorded_bytes
            st.session_state.recorded_bytes = None
            run_pipeline(data, file_ext="wav")
            st.rerun()

    # ── File Upload ───────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">Upload Audio</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "WAV / MP3 / M4A / OGG",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        key="audio_upload",
        label_visibility="collapsed",
    )
    if uploaded:
        st.audio(uploaded)
        if st.button("Transcribe & Run", key="btn_proc_upload", use_container_width=True, type="primary"):
            uploaded.seek(0)
            ext = Path(uploaded.name).suffix.lstrip(".")
            run_pipeline(uploaded.read(), file_ext=ext)
            st.rerun()

    # ── Output folder ─────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">Output Folder</p>', unsafe_allow_html=True)

    files = sorted(f for f in OUTPUT_DIR.rglob("*") if f.is_file())
    if files:
        for f in files:
            rel  = f.relative_to(OUTPUT_DIR)
            size = f.stat().st_size
            st.markdown(
                f'<span class="file-pill">{rel} <span style="opacity:.55">({size:,} B)</span></span>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("*(empty — agent-created files appear here)*")

    # ── Clear history ─────────────────────────────────────────────────────────
    st.divider()
    if st.button("Clear Chat History", key="btn_clear", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.pending_actions = []
        st.rerun()

    st.caption(f"Model: `{OLLAMA_MODEL}`")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CANVAS — Chat
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="agent-topbar">
  <div>
    <h2>Voice AI Agent</h2>
    <p>Audio in &rarr; intent classified &rarr; local tools executed</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; margin-top:4rem; color:rgba(255,255,255,0.3);">
      <p style="font-size:1rem; font-weight:500;">Record audio or type a command below to get started.</p>
      <p style="font-size:0.82rem; margin-top:0.5rem;">
        Try: <em>"Summarize this: ..."</em> &nbsp;|&nbsp;
        <em>"Write a Fibonacci function in Python"</em> &nbsp;|&nbsp;
        <em>"What is LangChain?"</em>
      </p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Collapsible pipeline log (assistant messages only)
            if msg["role"] == "assistant" and msg.get("pipeline"):
                with st.expander("View Execution Steps", expanded=False):
                    rows_html = ""
                    for step in msg["pipeline"]:
                        content = (
                            step["content"]
                            .replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                            .replace("\n", "<br>")
                        )
                        rows_html += f"""
                        <div class="pipeline-row">
                          <span class="pill {step['badge']}">{step['label']}</span>
                          <span>{content}</span>
                        </div>"""
                    st.markdown(rows_html, unsafe_allow_html=True)

# ── Pending actions — Approve / Deny ─────────────────────────────────────────
if st.session_state.pending_actions:
    st.markdown("---")
    st.warning("**File operations pending your approval:**")

    to_remove: list[int] = []

    for idx, action in enumerate(st.session_state.pending_actions):
        intent = action.get("intent", "?")
        fname  = action.get("filename", "?")
        desc   = action.get("description", "")

        st.markdown(f"""
        <div class="pending-card">
          <p>
            <strong>Action {idx+1}:</strong>
            <span class="pill pill-intent">{intent}</span>
            → <code>{fname}</code>
            {"<br><em style='font-size:0.8rem;opacity:.7;'>" + desc[:120] + "</em>" if desc else ""}
          </p>
        </div>
        """, unsafe_allow_html=True)

        a_col, d_col, _ = st.columns([1, 1, 4])
        with a_col:
            if st.button("Approve", key=f"approve_{idx}", use_container_width=True, type="primary"):
                pipeline = [
                    {"badge": "pill-action", "label": "Approved", "content": f"{intent} -> {fname}"},
                ]
                result = execute_action(action)
                pipeline.append({"badge": "pill-result", "label": "Result", "content": result[:400]})
                _add_message("assistant", result, pipeline)
                to_remove.append(idx)
                st.rerun()
        with d_col:
            if st.button("Deny", key=f"deny_{idx}", use_container_width=True):
                _add_message("assistant", f"[DENIED] {intent} on {fname}", [
                    {"badge": "pill-error", "label": "Denied", "content": f"User denied {intent} -> {fname}"}
                ])
                to_remove.append(idx)
                st.rerun()

    for i in sorted(to_remove, reverse=True):
        st.session_state.pending_actions.pop(i)

# ── Pinned text input ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Type a command or question…"):
    _add_message("user", prompt, [])

    pipeline: list[dict] = [
        {"badge": "pill-transcript", "label": "Text Input", "content": prompt},
    ]

    with st.spinner("Classifying intent..."):
        actions = classify(prompt)

    pipeline.append({
        "badge": "pill-intent",
        "label": "Intent(s)",
        "content": ", ".join(a.get("intent", "?") for a in actions),
    })

    results: list[str] = []
    for action in actions:
        intent = action.get("intent", "?")
        if intent in HUMAN_IN_LOOP:
            st.session_state.pending_actions.append(action)
            pipeline.append({
                "badge": "pill-intent",
                "label": "[PENDING]",
                "content": f"{intent} -> {action.get('filename','?')} (awaiting approval)",
            })
        else:
            pipeline.append({"badge": "pill-action", "label": "Action", "content": f"Executing: {intent}"})
            result = execute_action(action)
            pipeline.append({"badge": "pill-result", "label": "Result", "content": result[:400]})
            results.append(result)

    combined = "\n\n---\n\n".join(results) if results else "Awaiting your approval in the panel above."
    _add_message("assistant", combined, pipeline)
    st.rerun()
