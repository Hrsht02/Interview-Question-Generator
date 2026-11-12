import io
import logging
import json
import os
import re
import time
from collections import Counter
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="Prepwise — Interview Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.7").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
RETRIEVAL_K = 5
MAX_UPLOAD_MB = 20
MAX_RETRIES = 2

logger = logging.getLogger("interview_generator")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#172033; --muted:#697386; --line:#e7eaf0; --accent:#635bff; --soft:#f6f7fb; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background: #fbfcff; }
.block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background:rgba(251,252,255,.9); }
#MainMenu, footer { visibility:hidden; }
.hero { padding: 1.35rem 0 1.5rem; }
.eyebrow { color:var(--accent); font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { font-family:Manrope,sans-serif; font-size:clamp(2rem,4vw,3rem); line-height:1.12; letter-spacing:-.045em; margin:.55rem 0 .65rem; color:#172033; }
.hero p { color:var(--muted); font-size:1.02rem; max-width:700px; margin:0; }
.section-card { padding:0; }
.section-kicker { color:var(--accent); font-weight:700; font-size:.75rem; letter-spacing:.08em; text-transform:uppercase; }
.section-title { font-family:Manrope,sans-serif; font-size:1.2rem; font-weight:800; margin:.35rem 0 .25rem; }
.section-help { color:var(--muted); font-size:.88rem; margin-bottom:1rem; }
div[data-testid="stFileUploader"] section { border:1.5px dashed #cbd0df; border-radius:13px; background:#fafbff; }
div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-radius:11px; background:#fafbff; }
div.stButton > button[kind="primary"] { background:var(--accent); border:0; border-radius:8px; min-height:2.55rem; padding:.4rem 1rem; font-weight:600; box-shadow:none; }
div.stButton > button[kind="primary"]:hover { background:#5148ed; border:0; }
div.stButton > button { border-radius:8px; }
.metric-strip { background:#f4f5f8; border:1px solid #e7e9ef; border-radius:8px; padding:.65rem .8rem; color:#465064; font-size:.88rem; }
.result-head { font-family:Manrope,sans-serif; font-size:1.55rem; font-weight:800; letter-spacing:-.025em; }
.question-card { background:white; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; margin:.65rem 0; }
.answer-label { color:var(--accent); font-weight:700; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:white; }
.small-note { color:#8790a1; font-size:.78rem; }
@media (max-width: 800px) {
 .block-container { padding:1rem 1rem 3rem; }
 .section-card { padding:1rem; }
}
</style>
""", unsafe_allow_html=True)


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(parts)
    if name.endswith((".txt", ".md", ".markdown")):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_documents(files) -> List[Document]:
    docs = []
    remaining_chars = 250_000
    for file in files:
        raw = clean_text(extract_text(file))
        if not raw:
            continue
        if len(raw) > remaining_chars:
            raw = raw[:remaining_chars]
        remaining_chars -= len(raw)
        if remaining_chars < 0:
            remaining_chars = 0
        docs.append(Document(page_content=raw, metadata={"source": file.name}))
        if remaining_chars == 0:
            break
    if not docs:
        raise ValueError("No readable text was found. Scanned PDFs need OCR before upload.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=80, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    chunks = chunks[:300]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def retrieve_context(chunks, topic: str, k: int = RETRIEVAL_K):
    """Keep the existing lightweight lexical RAG retrieval and ranking strategy."""
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "into", "are", "was",
        "were", "you", "your", "what", "when", "where", "which", "how", "why",
        "about", "using", "use", "can", "could", "should", "would", "have", "has",
        "had", "not", "but", "all", "any", "its", "their", "they", "them", "then",
        "than", "also", "each", "such", "these", "those", "our", "out", "more"
    }
    terms = [
        t for t in re.findall(r"[a-zA-Z0-9_+#.-]{2,}", (topic or "").lower())
        if t not in stop_words
    ]
    if not terms:
        return chunks[:min(k, len(chunks))]
    scored = []
    for index, chunk in enumerate(chunks):
        words = re.findall(r"[a-zA-Z0-9_+#.-]{2,}", chunk.page_content.lower())
        counts = Counter(words)
        score = sum(min(counts.get(term, 0), 4) for term in terms)
        scored.append((score, -index, chunk))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    matches = [item[2] for item in scored[:k] if item[0] > 0]
    return matches or chunks[:min(k, len(chunks))]


def build_gemini_llm():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Gemini fallback is not configured. Add GEMINI_API_KEY in Render.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        max_retries=0,
        timeout=60,
    )


def _extract_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    pieces = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                pieces.append(value)
    return "\n".join(pieces).strip()


def _is_transient(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(code in message for code in (
        "429", "500", "502", "503", "504", "rate_limit",
        "overloaded", "temporarily", "timeout", "connection"
    ))


def invoke_with_resilience(prompt: str) -> str:
    """Grok is primary; Gemini is invoked as a fallback after bounded retries."""
    grok_key = os.getenv("XAI_API_KEY", "").strip()
    primary_error = None

    if grok_key:
        client = OpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1",
            timeout=90.0,
            max_retries=0,
        )
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.responses.create(
                    model=GROK_MODEL,
                    input=prompt,
                    reasoning={"effort": "low"},
                    store=False,
                )
                generated = _extract_text(response)
                if not generated:
                    raise RuntimeError("Grok returned an empty response.")
                return generated
            except Exception as exc:
                primary_error = exc
                logger.warning(
                    "Grok primary request failed (attempt %s/%s): %s",
                    attempt + 1, MAX_RETRIES + 1, type(exc).__name__,
                )
                if not _is_transient(exc) or attempt >= MAX_RETRIES:
                    break
                time.sleep(1.0 * (2 ** attempt))
    else:
        primary_error = RuntimeError("XAI_API_KEY is missing.")

    try:
        logger.info("Attempting Gemini fallback after Grok primary failure.")
        response = build_gemini_llm().invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        generated = str(content).strip()
        if not generated:
            raise RuntimeError("Gemini returned an empty response.")
        return generated
    except Exception as fallback_error:
        logger.exception("Gemini fallback request failed")
        raise RuntimeError(
            "Both Grok and Gemini generation attempts failed. Check Render logs, "
            "provider API keys, model access, and available quota."
        ) from fallback_error


def parse_json_array(raw: str):
    cleaned = re.sub(r"^\s*```(?:json)?|\`\`\`\s*$", "", raw.strip(), flags=re.I)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("The AI response was not in the expected format. Please retry.")
    data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("The AI response had an unexpected format.")
    return data


def generate_interview_set(chunks, topic: str, role: str, difficulty: str, count: int):
    """One model round-trip creates the questions and their answers to reduce latency."""
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'document')} | chunk {d.metadata.get('chunk_id', '?')}]\n"
        f"{d.page_content}"
        for d in chunks
    )
    prompt = f"""
You are an expert technical interviewer and interview coach.
Create exactly {count} distinct, useful interview question-and-answer pairs.
Topic/focus: {topic or "Choose the most relevant topics in the source"}
Target role: {role or "General technical role"}
Difficulty: {difficulty}

Use the retrieved source passages as the primary source of truth.
Questions must be grounded in the passages or clearly test a directly related concept.
Avoid duplicates. Cover different concepts. Do not claim the source says something it does not.
Answers should be interview-ready: direct opening, clear explanation, and a concise example
where useful. If the source is insufficient, say what is missing instead of fabricating.
Return ONLY a valid JSON array. Every object must contain:
"question", "concept", "answer", "source_hint".
No markdown fences and no text outside the JSON array.

RETRIEVED SOURCE CONTEXT:
{context}
"""
    parsed = parse_json_array(invoke_with_resilience(prompt))
    results = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("question") or not item.get("answer"):
            continue
        results.append({
            "question": str(item["question"]).strip(),
            "concept": str(item.get("concept", "")).strip(),
            "answer": str(item["answer"]).strip(),
            "source_hint": str(item.get("source_hint", "")).strip(),
            "sources": [
                {
                    "source": d.metadata.get("source", "document"),
                    "chunk_id": d.metadata.get("chunk_id", "?"),
                    "text": d.page_content,
                } for d in chunks
            ],
        })
    if not results:
        raise ValueError("No complete question-and-answer pairs were returned. Please retry.")
    return results[:count]


def main():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">AI-powered interview preparation</div>
      <h1>Turn your study material into<br>interview confidence.</h1>
      <p>Upload your notes or documentation. Get role-focused questions, clear answers,
      and source-backed explanations in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        st.markdown('<div class="section-kicker">Step 01 · Your material</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Add source documents</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-help">Use class notes, technical documentation, PDFs, or your own study material.</div>', unsafe_allow_html=True)
        files = st.file_uploader(
            "Upload PDF, DOCX, TXT or Markdown",
            type=["pdf", "docx", "txt", "md", "markdown"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if files:
            total_mb = sum(f.size for f in files) / (1024 * 1024)
            st.markdown(
                f'<div class="metric-strip">✓ {len(files)} file(s) ready · {total_mb:.1f} MB uploaded</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Text-based PDFs are supported. Scanned/image-only PDFs need OCR first.")

    with right:
        st.markdown('<div class="section-kicker">Step 02 · Interview setup</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Personalise your practice</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-help">Choose what you are preparing for. Retrieval settings are optimised automatically.</div>', unsafe_allow_html=True)
        topic = st.text_input("Topic or focus", placeholder="e.g. Python, SQL, RAG, system design")
        role = st.text_input("Target role", placeholder="e.g. Data Analyst, GenAI Developer")
        difficulty = st.selectbox("Question difficulty", ["Beginner", "Intermediate", "Advanced"], index=1)
        count = st.select_slider("Number of questions", options=[3, 5, 7, 10], value=5)

    st.write("")
    _, action_col, _ = st.columns([1, 1.35, 1])
    with action_col:
        generate = st.button(
            "Generate questions & answers",
            type="primary",
            use_container_width=False,
            disabled=not bool(files),
        )
    if not files:
        st.caption("Upload at least one source document to start generating.")

    if generate:
        if any(f.size > MAX_UPLOAD_MB * 1024 * 1024 for f in files):
            st.error(f"Each uploaded file must be {MAX_UPLOAD_MB} MB or smaller.")
        else:
            try:
                with st.status("Preparing your interview set…", expanded=True) as status:
                    st.write("Extracting text and building source chunks")
                    chunks = make_documents(files)
                    st.write(f"Retrieving the most relevant passages from {len(chunks)} chunks")
                    retrieved = retrieve_context(chunks, topic, RETRIEVAL_K)
                    st.write("Generating questions and answers")
                    results = generate_interview_set(
                        retrieved, topic, role, difficulty, int(count)
                    )
                    status.update(label="Your interview set is ready", state="complete", expanded=False)
                st.session_state["results"] = results
                st.session_state["run_meta"] = {
                    "files": [f.name for f in files],
                    "chunks": len(chunks),
                    "retrieved": len(retrieved),
                    "topic": topic or "Document topics",
                    "role": role or "General technical role",
                    "difficulty": difficulty,
                }
                st.success(f"Generated {len(results)} interview question-and-answer pairs.")
            except Exception as exc:
                # Keep the complete provider traceback in Render logs for diagnosis,
                # while showing a safe, actionable message in the browser.
                logger.exception("Interview question generation failed")
                detail = str(exc).strip()
                st.error(detail or f"Generation failed ({type(exc).__name__}).")
                st.caption(
                    f"Technical detail: {type(exc).__name__}. "
                    "For the exact provider response, open Render → your service → Logs "
                    "and inspect the latest 'Interview question generation failed' entry. "
                    "Do not share your API key or environment-variable values."
                )

    if "results" in st.session_state:
        st.write("")
        st.markdown('<div class="result-head">Your interview practice set</div>', unsafe_allow_html=True)
        meta = st.session_state.get("run_meta", {})
        st.caption(
            f"{meta.get('role', 'Technical role')} · {meta.get('topic', 'All topics')} · "
            f"{meta.get('difficulty', 'Intermediate')} · "
            f"{meta.get('retrieved', 0)} relevant passages"
        )
        export = []
        for idx, item in enumerate(st.session_state["results"], start=1):
            with st.expander(f"Q{idx:02d}  ·  {item['question']}", expanded=(idx == 1)):
                if item.get("concept"):
                    st.caption(f"CONCEPT  ·  {item['concept']}")
                st.markdown('<div class="answer-label">Suggested interview answer</div>', unsafe_allow_html=True)
                st.markdown(item["answer"])
                if item.get("source_hint"):
                    st.markdown("**Why this question is relevant**")
                    st.caption(item["source_hint"])
                sources = item.get("sources", [])
                if sources:
                    with st.expander("View retrieved source passages"):
                        seen = set()
                        for source in sources:
                            key = (source["source"], source["chunk_id"])
                            if key in seen:
                                continue
                            seen.add(key)
                            st.markdown(f"**{source['source']} · Chunk {source['chunk_id']}**")
                            st.write(source["text"])
            export.append(
                f"## Question {idx}\n{item['question']}\n\n### Suggested answer\n{item['answer']}\n"
                + (f"\n**Concept:** {item['concept']}\n" if item.get("concept") else "")
            )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.download_button(
                "↓  Download interview set (.md)",
                data="\n\n---\n\n".join(export),
                file_name="interview_practice_set.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_b:
            if st.button("↻  Start a new practice set", use_container_width=True):
                st.session_state.pop("results", None)
                st.session_state.pop("run_meta", None)
                st.rerun()

    st.divider()
    st.markdown(
        '<div class="small-note">Privacy & accuracy · Documents are processed for this session. '
        'Relevant text is sent to Grok, with Gemini as an automatic fallback, to generate answers. '
        'Review generated answers before using them in an interview.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
