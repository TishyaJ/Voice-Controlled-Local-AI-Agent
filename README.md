# 🎙️ Voice-Controlled Local AI Agent

A production-ready voice agent that transcribes speech, classifies intent, and executes local tools — all from a premium split-screen Streamlit UI.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/STT-Groq_Whisper-orange)](https://console.groq.com)
[![Ollama](https://img.shields.io/badge/LLM-Ollama_qwen-green)](https://ollama.ai)

---

## 📐 Architecture

```
          ┌──────────────────────────────────┐
          │        Streamlit UI (app.py)      │
          │  Left (60%) │  Right (40%)        │
          │  Controls   │  Chat History        │
          └──────┬───────────────────────────┘
                 │
         ┌───────▼────────┐
         │ audio_processor│  ──►  Groq Whisper API  ──►  transcript
         └───────┬────────┘
                 │
         ┌───────▼────────────────────┐
         │  LangChain + Ollama (qwen) │  ──►  JSON array of actions
         └───────┬────────────────────┘
                 │
         ┌───────▼────────┐
         │   tools.py     │  create_file | write_code | summarize | chat
         └───────┬────────┘
                 │
           output/ (sandboxed)
```

### Key Components

| File | Responsibility |
|------|---------------|
| `app.py` | Streamlit UI, LangChain orchestration, Human-in-the-Loop, compound command loop |
| `audio_processor.py` | Groq Whisper transcription, mic recording, file upload support |
| `tools.py` | Tool implementations with output/ directory safety constraint |
| `.env` | API keys and model configuration |

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running: [https://ollama.ai](https://ollama.ai)
- **Groq API key** (free): [https://console.groq.com](https://console.groq.com)
- A working microphone (for mic input)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/voice-ai-agent.git
cd voice-ai-agent

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Pull the Ollama Model

```bash
ollama pull qwen2.5:3b
# If you prefer a different tag:
# ollama pull qwen:4b
```

### 3. Configure Environment

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
OLLAMA_MODEL=qwen2.5:3b
```

### 4. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎯 Supported Intents

| Intent | Trigger phrase examples | Tool |
|--------|------------------------|------|
| **General Chat** | "What is machine learning?", "Tell me a joke" | `general_chat()` |
| **Summarize Text** | "Summarize this: …", "Give me a summary of …" | `summarize_text()` |
| **Create File** | "Create a file called notes.txt" | `create_file()` ⚠️ Requires approval |
| **Write Code** | "Write a Python retry function and save it to retry.py" | `generate_and_write_code()` ⚠️ Requires approval |

### Compound Commands (Bonus ✅)
> "Summarize this text and save it to summary.txt"

The LLM returns a JSON array and both actions are executed sequentially.

---

## 🔒 Safety Constraint

All file operations are **strictly sandboxed** to `./output/` using `os.path.realpath()` validation. Any path traversal attempt (e.g., `../../secret.txt`) is rejected before reaching the filesystem.

---

## ✨ Bonus Features

| Feature | Status | Implementation |
|---------|--------|---------------|
| Compound Commands | ✅ | LLM outputs JSON array; app loops over each action |
| Human-in-the-Loop | ✅ | `create_file` + `write_code` show Approve/Deny buttons |
| Graceful Degradation | ✅ | All exceptions → fallback `general_chat` |
| Memory | ✅ | `StreamlitChatMessageHistory` + `st.session_state` |

---

## 🛠️ Hardware Note (STT Choice)

Running `openai/whisper-large-v3` locally on a **CPU-only Windows machine** requires:
- ~4 GB RAM for model loading
- 30–120 seconds transcription time per utterance

This makes the UX unusable. Instead, we use **Groq's hosted Whisper API** which delivers:
- Sub-second transcription
- Free tier (generous limits)
- Zero local CPU usage (keeps CPU free for Ollama)

This trade-off is intentional and documented here per the assignment requirements.

---

## 📁 Project Structure

```
voice-ai-agent/
├── app.py                 # Main Streamlit application
├── audio_processor.py     # STT via Groq Whisper
├── tools.py               # Tool implementations
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── .gitignore
├── README.md
└── output/                # ← All agent-created files land here (gitignored)
```

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `GROQ_API_KEY not found` | Make sure `.env` exists and is populated |
| `Connection refused` (Ollama) | Run `ollama serve` in a separate terminal |
| `sounddevice` error | Install PortAudio: `winget install -e --id PortAudio.PortAudio` or use file upload |
| Model not found | Run `ollama pull qwen2.5:3b` (or your chosen tag) |

---

## 📜 License

MIT
