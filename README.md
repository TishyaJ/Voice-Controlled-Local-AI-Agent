# 🎙️ Voice-Controlled Local AI Agent

A production-ready voice agent that transcribes speech, classifies intent, and executes local tools using a modern, chat-first UI. Built specifically for CPU-only Windows environments.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/STT-Groq_Whisper-orange)](https://console.groq.com)
[![Ollama](https://img.shields.io/badge/LLM-Ollama_qwen2.5-green)](https://ollama.ai)

---

## 📐 Architecture Explanation

The system is designed with a **"Chat-First" UI layout** that mimics modern AI clients like ChatGPT or Claude, keeping the technical complexity hidden but accessible.

```text
          ┌──────────────────────────────────┐
          │        Streamlit UI (app.py)      │
          │  Sidebar:      │ Main Canvas:     │
          │  Mic / Upload  │ Chat Bubbles &   │
          │  Output Files  │ Pinned Input     │
          └──────┬─────────┴─────────┬───────┘
                 │                   │
         ┌───────▼────────┐          │
         │ audio_processor│          │
         │ (Groq Whisper) │          │
         └───────┬────────┘          │
                 │transcript         │
         ┌───────▼───────────────────▼┐
         │  LangChain + Ollama (Local)│  ──► JSON Action Array
         └───────┬────────────────────┘
                 │
         ┌───────▼────────┐
         │   tools.py     │  create_file | write_code | summarize | chat
         └───────┬────────┘
                 │
           output/ (sandboxed)
```

**Workflow:**
1. **Input:** Audio is captured via `sounddevice` or uploaded, then sent to `audio_processor.py`.
2. **STT:** Audio is converted to text.
3. **Brain:** The text along with conversation history is sent to a local Ollama model (`qwen2.5:1.5b`). The prompt forces the model to output a strictly formatted JSON array of actions.
4. **Execution:** The JSON is parsed by `app.py`. Safe actions execute immediately. Dangerous actions (`create_file`, `write_code`) trigger a "Human-in-the-Loop" pause, rendering Approve/Deny buttons inline in the chat.
5. **Output:** Results are presented in native chat bubbles. Technical execution steps are tucked inside a `⚙️ View Execution Steps` expander to keep the UI clean.

---

## 🛠️ Hardware Workarounds Used

This project was built on a **CPU-only Windows machine**. To achieve real-time usability, two major architectural trade-offs were made:

### 1. STT: Groq API instead of Local Whisper
Running `openai/whisper-large-v3` locally purely on a CPU requires ~4 GB RAM and takes **30–120 seconds** to transcribe a 5-second utterance. This makes a voice UI completely unusable. 
* **Workaround:** We utilize the free **Groq Whisper API endpoint**. This delivers sub-second transcription and offloads the heavy processing, keeping the local CPU completely free for the LLM inference.

### 2. LLM: `qwen2.5:1.5b` instead of larger models
Running a 7B+ or 8B+ parameter model (like Llama 3 8B) on CPU results in agonizingly slow token generation. Even a 3B-4B Vision model proved too slow.
* **Workaround:** We explicitly target **`qwen2.5:1.5b`**. At under 1GB of memory, it is incredibly fast on CPU (generating responses in 2-4 seconds) while still being highly capable of strict JSON instruction-following. We further optimized this in Python by setting `num_predict=256` and `num_ctx=2048` to prevent the CPU from churning on unnecessary context.

---

## ⚡ Setup Instructions

### Prerequisites
- **Python 3.10+** installed
- **Ollama** installed and running: [Download here](https://ollama.ai)
- **Groq API key** (free): [Get one here](https://console.groq.com)
- A working microphone

### 1. Clone & Install Environment
```powershell
git clone https://github.com/TishyaJ/Voice-Controlled-Local-AI-Agent.git
cd Voice-Controlled-Local-AI-Agent

# It is highly recommended to use a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt
```

### 2. Pull the Local AI Model
Ensure the Ollama app is running on your machine, then open a terminal and pull the lightweight Qwen model:
```powershell
ollama pull qwen2.5:1.5b
```

### 3. Configure the `.env` File
Copy the example environment file:
```powershell
copy .env.example .env
```
Open `.env` in a text editor and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_key_here
OLLAMA_MODEL=qwen2.5:1.5b
```

### 4. Run the Agent
Make sure your virtual environment is activated, then launch Streamlit:
```powershell
python -m streamlit run app.py
```
*The UI will automatically open in your browser at `http://localhost:8501`.*

---

## 🎯 Supported Features & Intents

### Capabilities
- **General Chat:** Normal conversational AI (`general_chat()` tool).
- **Summarization:** "Summarize this long text: ..." (`summarize_text()` tool).
- **File Creation:** "Make a new file called data.csv" (`create_file()` tool).
- **Code Generation:** "Write a Python retry loop and save it to retry.py" (`generate_and_write_code()` tool).

### ✨ Assignment Bonus Features Implemented
* ✅ **Compound Commands:** Say *"Summarize this and save it to summary.txt"*. The LLM outputs an array of JSON objects, and the system executes them sequentially.
* ✅ **Human-in-the-Loop:** Before ANY file is created or written to, the chat UI halts and presents "Approve" / "Deny" buttons.
* ✅ **Graceful Degradation:** If the LLM completely fails to output JSON, or the audio is garbled, the system automatically catches the exception and falls back to a polite conversational response.
* ✅ **Memory:** Streamlit's session state ensures the entire conversation history context is passed back to the LLM on every turn.

---

## 🔒 Safety Constraint

All file operations are **strictly sandboxed** to the `./output/` directory within the project folder. No matter what prompt the AI generates, `os.path.realpath()` validation guarantees it cannot escape `output/`. A request for `../../windows/system32/secret.txt` will fail safely.
