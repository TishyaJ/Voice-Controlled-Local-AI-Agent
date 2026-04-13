"""
tools.py — Tool implementations for the Voice-Controlled Local AI Agent.

SAFETY CONSTRAINT: Every file-system operation is restricted to the OUTPUT_DIR.
Any attempt to escape the sandbox (via path traversal or absolute paths) is
silently rejected and returns an error string.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Optional

from langchain_ollama import OllamaLLM

# ── Sandbox directory ────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path(os.getcwd()) / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ollama model (shared across tools) ──────────────────────────────────────
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")  # adjust tag as needed


def _safe_path(filename: str) -> Optional[Path]:
    """
    Resolve *filename* inside OUTPUT_DIR and verify it doesn't escape the sandbox.

    Returns the resolved Path on success, or None if the path is unsafe.
    """
    # Strip leading slashes / drive letters that would make it absolute
    clean = re.sub(r"^[/\\]+", "", filename.strip())
    # Resolve relative to OUTPUT_DIR
    candidate = (OUTPUT_DIR / clean).resolve()
    try:
        candidate.relative_to(OUTPUT_DIR.resolve())
        return candidate
    except ValueError:
        return None  # path escapes sandbox


# ── Tool 1: Create a file ────────────────────────────────────────────────────

def create_file(filename: str) -> str:
    """Create an empty file inside the output/ sandbox."""
    path = _safe_path(filename)
    if path is None:
        return f"❌ Safety violation: '{filename}' resolves outside the output/ directory."

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return f"ℹ️  File already exists: `output/{path.relative_to(OUTPUT_DIR)}`"

    path.touch()
    return f"✅ Created empty file: `output/{path.relative_to(OUTPUT_DIR)}`"


# ── Tool 2: Write code ───────────────────────────────────────────────────────

def write_code(filename: str, code: str) -> str:
    """Write *code* to *filename* inside the output/ sandbox."""
    path = _safe_path(filename)
    if path is None:
        return f"❌ Safety violation: '{filename}' resolves outside the output/ directory."

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(code), encoding="utf-8")
    line_count = len(code.splitlines())
    return (
        f"✅ Wrote {line_count} lines of code to "
        f"`output/{path.relative_to(OUTPUT_DIR)}`"
    )


# ── Tool 3: Summarize text ───────────────────────────────────────────────────

def summarize_text(text: str, save_to: Optional[str] = None) -> str:
    """
    Use the local Ollama model to summarize *text*.
    Optionally save the summary to *save_to* (inside output/).
    """
    if not text.strip():
        return "❌ No text provided to summarize."

    llm = OllamaLLM(model=_OLLAMA_MODEL, temperature=0.3)
    prompt = (
        "Please provide a concise, well-structured summary of the following text. "
        "Use bullet points where appropriate.\n\n"
        f"TEXT:\n{text}\n\nSUMMARY:"
    )
    try:
        summary = llm.invoke(prompt)
    except Exception as exc:
        return f"❌ Summarization failed: {exc}"

    result = f"📝 **Summary:**\n\n{summary}"

    if save_to:
        path = _safe_path(save_to)
        if path is None:
            result += f"\n\n❌ Could not save — unsafe path: '{save_to}'"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(summary, encoding="utf-8")
            result += f"\n\n✅ Summary saved to `output/{path.relative_to(OUTPUT_DIR)}`"

    return result


# ── Tool 4: General chat ─────────────────────────────────────────────────────

def general_chat(message: str, history: list[dict] | None = None) -> str:
    """
    Standard conversational response using the local Ollama model.
    *history* is a list of {"role": "human"|"ai", "content": str} dicts.
    """
    llm = OllamaLLM(model=_OLLAMA_MODEL, temperature=0.7)

    # Build a simple context string from history
    context_parts: list[str] = []
    for turn in (history or [])[-6:]:  # last 6 turns for brevity
        role = "User" if turn["role"] == "human" else "Assistant"
        context_parts.append(f"{role}: {turn['content']}")
    context = "\n".join(context_parts)

    prompt = (
        "You are a helpful AI assistant. Answer the user's message clearly and concisely.\n\n"
        + (f"Conversation so far:\n{context}\n\n" if context else "")
        + f"User: {message}\nAssistant:"
    )

    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as exc:
        return f"❌ Chat failed: {exc}"


# ── Tool 5: Generate code with LLM then write ────────────────────────────────

def generate_and_write_code(description: str, filename: str) -> str:
    """
    Use the local Ollama model to generate code from *description*,
    then write it to *filename* inside output/.
    """
    llm = OllamaLLM(model=_OLLAMA_MODEL, temperature=0.2)
    prompt = (
        "You are an expert programmer. Write clean, well-commented, production-ready code "
        "based on the following description. Output ONLY the raw code — no markdown fences, "
        "no explanations.\n\n"
        f"Description: {description}\n\nCode:"
    )

    try:
        generated_code = llm.invoke(prompt)
    except Exception as exc:
        return f"❌ Code generation failed: {exc}"

    # Strip markdown fences if the model included them anyway
    generated_code = re.sub(r"^```[a-z]*\n?", "", generated_code, flags=re.MULTILINE)
    generated_code = re.sub(r"```$", "", generated_code, flags=re.MULTILINE).strip()

    return write_code(filename, generated_code) + f"\n\n```\n{generated_code[:800]}\n```"


# ── Dispatcher (used by app.py) ──────────────────────────────────────────────

TOOL_MAP = {
    "create_file": create_file,
    "write_code": write_code,
    "generate_and_write_code": generate_and_write_code,
    "summarize_text": summarize_text,
    "general_chat": general_chat,
}
