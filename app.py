import io
import logging
import json
import os
import re
import time
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
.block-container { max-width: 1240px; padding-top: .65rem; padding-bottom: 1.5rem; }
header[data-testid="stHeader"] { background:rgba(251,252,255,.9); }
#MainMenu, footer { visibility:hidden; }
.hero { padding: .35rem 0 .7rem; }
.eyebrow { color:var(--accent); font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { font-family:Manrope,sans-serif; font-size:clamp(1.9rem,3.1vw,2.45rem); line-height:1.08; letter-spacing:-.04em; margin:.35rem 0 .45rem; color:#172033; }
.hero p { color:var(--muted); font-size:.92rem; max-width:700px; margin:0; }
.section-card { padding:0; }
.section-kicker { color:var(--accent); font-weight:700; font-size:.75rem; letter-spacing:.08em; text-transform:uppercase; }
.section-title { font-family:Manrope,sans-serif; font-size:1.08rem; font-weight:700; margin:.22rem 0 .2rem; }
.section-help { color:var(--muted); font-size:.82rem; margin-bottom:.55rem; }
div[data-testid="stFileUploader"] section { border:1.5px dashed #cbd0df; border-radius:13px; background:#fafbff; }
div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-radius:8px; background:#fff; min-height:2.35rem; }
div[data-testid="stTextInput"], div[data-testid="stSelectbox"], div[data-testid="stSlider"] { margin-bottom:.35rem; }\ndiv[data-testid="stFileUploader"] { margin-bottom:.35rem; }
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
 .block-container { padding:.65rem 1rem 1.5rem; }
 .section-card { padding:0; }
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
    for source_index, file in enumerate(files):
        raw = clean_text(extract_text(file))
        if not raw:
            continue
        if len(raw) > remaining_chars:
            raw = raw[:remaining_chars]
        remaining_chars -= len(raw)
        if remaining_chars < 0:
            remaining_chars = 0
        docs.append(Document(page_content=raw, metadata={"source": file.name, "source_index": source_index}))
        if remaining_chars == 0:
            break
    if not docs:
        raise ValueError("No readable text was found. Scanned PDFs need OCR before upload.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=80, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    chunks = chunks[:300]
    per_source_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        chunk_id = per_source_counts.get(source, 0)
        chunk.metadata["chunk_id"] = chunk_id
        per_source_counts[source] = chunk_id + 1
    return chunks


def build_vector_store(chunks):
    """Create an in-memory FAISS index using Gemini text embeddings."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required for document embeddings and Gemini fallback. "
            "Add it in Render Environment."
        )
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()
    document_embeddings = GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
        task_type="RETRIEVAL_DOCUMENT",
    )
    query_embeddings = GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
        task_type="RETRIEVAL_QUERY",
    )
    return FAISS.from_documents(chunks, document_embeddings), query_embeddings


def retrieve_context(vector_store, query_embeddings, topic: str, role: str, k: int = RETRIEVAL_K):
    """Retrieve semantically relevant chunks with FAISS similarity search."""
    query = " ".join(part for part in [topic.strip(), role.strip()] if part)
    if not query:
        query = "Core concepts, practical knowledge, and interview topics in the uploaded documents"
    query_vector = query_embeddings.embed_query(query)
    matches = vector_store.similarity_search_by_vector(query_vector, k=k)
    return sorted(
        matches,
        key=lambda chunk: (
            chunk.metadata.get("source_index", 0),
            chunk.metadata.get("start_index", 0),
            chunk.metadata.get("chunk_id", 0),
        ),
    )


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


def generate_interview_set(chunks, topic: str, role: str, difficulty: str, count: int, progress_callback=None):
    """Two-stage generation: first create questions, then generate grounded answers."""
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'document')} | chunk {d.metadata.get('chunk_id', '?')}]\n"
        f"{d.page_content}"
        for d in chunks
    )

    if progress_callback:
        progress_callback("Stage 1/2 · Generating interview questions")
    question_prompt = f"""
You are an expert technical interviewer.
Generate exactly {count} distinct interview questions.
Topic/focus: {topic or "Relevant topics from the uploaded material"}
Target role: {role or "General technical role"}
Difficulty: {difficulty}

Use the retrieved source context as the factual basis. Questions should test understanding
and practical application, fit the target role, and avoid duplicates.
Return ONLY a valid JSON array of objects with fields "question", "concept", "source_hint".
Do not include answers in this stage. No markdown fences or surrounding commentary.

RETRIEVED SOURCE CONTEXT:
{context}
"""
    question_items = parse_json_array(invoke_with_resilience(question_prompt))
    questions = [
        {
            "question": str(item.get("question", "")).strip(),
            "concept": str(item.get("concept", "")).strip(),
            "source_hint": str(item.get("source_hint", "")).strip(),
        }
        for item in question_items
        if isinstance(item, dict) and item.get("question")
    ][:count]
    if not questions:
        raise ValueError("The question-generation stage returned no usable questions.")

    if progress_callback:
        progress_callback("Stage 2/2 · Generating grounded answers")
    answer_prompt = f"""
You are an expert technical interview coach.
Write an accurate, interview-ready answer for each question below.
Target role: {role or "General technical role"}
Difficulty: {difficulty}

Use the retrieved context as the primary source of truth. Explain clearly and add a concise
practical example where useful. Do not invent facts absent from the context. If the context
is insufficient, state that explicitly. Keep each question text unchanged.

QUESTIONS:
{json.dumps(questions, ensure_ascii=False)}

Return ONLY a valid JSON array with one object per question and fields "question", "answer".
No markdown fences and no text outside the JSON array.

RETRIEVED SOURCE CONTEXT:
{context}
"""
    answer_items = parse_json_array(invoke_with_resilience(answer_prompt))
    answers_by_question = {
        str(item.get("question", "")).strip(): str(item.get("answer", "")).strip()
        for item in answer_items
        if isinstance(item, dict) and item.get("question") and item.get("answer")
    }

    results = []
    for index, item in enumerate(questions):
        answer = answers_by_question.get(item["question"])
        if not answer and index < len(answer_items) and isinstance(answer_items[index], dict):
            answer = str(answer_items[index].get("answer", "")).strip()
        if answer:
            results.append({
                **item,
                "answer": answer,
                "sources": [
                    {
                        "source": doc.metadata.get("source", "document"),
                        "chunk_id": doc.metadata.get("chunk_id", "?"),
                        "text": doc.page_content,
                    }
                    for doc in chunks
                ],
            })
    if not results:
        raise ValueError("The answer-generation stage returned no usable answers.")
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
            total_bytes = sum(f.size for f in files)
            size_label = f"{total_bytes / 1024:.0f} KB" if total_bytes < 1024 * 1024 else f"{total_bytes / (1024 * 1024):.1f} MB"
            st.markdown(
                f'<div class="metric-strip">✓ {len(files)} file(s) ready · {size_label} uploaded</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Text-based PDFs are supported. Scanned/image-only PDFs need OCR first.")
        generate = st.button(
            "Generate questions & answers",
            type="primary",
            disabled=not bool(files),
        )

    with right:
        st.markdown('<div class="section-kicker">Step 02 · Interview setup</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Personalise your practice</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-help">Choose your focus. FAISS semantic search finds the most relevant passages.</div>', unsafe_allow_html=True)
        topic = st.text_input("Topic or focus", placeholder="e.g. Python, SQL, RAG, system design")
        role = st.text_input("Target role", placeholder="e.g. Data Analyst, GenAI Developer")
        difficulty = st.selectbox("Question difficulty", ["Beginner", "Intermediate", "Advanced"], index=1)
        count = st.select_slider("Number of questions", options=[3, 5, 7, 10], value=5)

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
                    st.write("Creating Gemini embeddings and building the FAISS index")
                    vector_store, query_embeddings = build_vector_store(chunks)
                    st.write(f"Running FAISS semantic search across {len(chunks)} chunks")
                    retrieved = retrieve_context(
                        vector_store, query_embeddings, topic, role, RETRIEVAL_K
                    )
                    results = generate_interview_set(
                        retrieved, topic, role, difficulty, int(count),
                        progress_callback=status.write,
                    )
                    status.update(label="Your interview set is ready", state="complete", expanded=False)
                st.session_state["results"] = results
                st.session_state["all_chunks"] = [
                    {
                        "source": chunk.metadata.get("source", "document"),
                        "source_index": chunk.metadata.get("source_index", 0),
                        "chunk_id": chunk.metadata.get("chunk_id", 0),
                        "text": chunk.page_content,
                    }
                    for chunk in chunks
                ]
                st.session_state["retrieved_chunk_keys"] = {
                    (chunk.metadata.get("source", "document"), chunk.metadata.get("chunk_id", 0))
                    for chunk in retrieved
                }
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
            export.append(
                f"## Question {idx}\n{item['question']}\n\n### Suggested answer\n{item['answer']}\n"
                + (f"\n**Concept:** {item['concept']}\n" if item.get("concept") else "")
            )
        all_chunks = st.session_state.get("all_chunks", [])
        retrieved_keys = st.session_state.get("retrieved_chunk_keys", set())
        if all_chunks:
            with st.expander(f"Source document chunks · {len(all_chunks)} total", expanded=False):
                st.caption("Chunks are displayed in original document order. Retrieved chunks are marked; other chunks were not selected for this generation.")
                for chunk in all_chunks:
                    key = (chunk["source"], chunk["chunk_id"])
                    is_retrieved = key in retrieved_keys
                    status = " · Retrieved" if is_retrieved else " · Not retrieved"
                    st.markdown(f"**{chunk['source']} · Chunk {chunk['chunk_id']}{status}**")
                    st.write(chunk["text"])
                    st.divider()

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
                st.session_state.pop("all_chunks", None)
                st.session_state.pop("retrieved_chunk_keys", None)
                st.rerun()

    st.divider()
    st.markdown(
        '<div class="small-note"><strong>How it works:</strong> Document extraction → Recursive chunking → Gemini embeddings → FAISS semantic search → Question prompt → Answer prompt → Interview-ready Q&A</div>'
        '<div class="small-note">Gemini creates embeddings · Grok generates questions and answers, with Gemini fallback · Documents and FAISS index are processed in memory for this session.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
