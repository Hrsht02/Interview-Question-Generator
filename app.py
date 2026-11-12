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
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(
    page_title="Prepwise — Interview Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
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
div[data-testid="stTextInput"], div[data-testid="stSelectbox"], div[data-testid="stSlider"] { margin-bottom:.35rem; }
div[data-testid="stFileUploader"] { margin-bottom:.35rem; }
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


def _error_summary(exc: Exception, limit: int = 900) -> str:
    """Return provider error details without logging request prompts or credentials."""
    status = getattr(exc, "status_code", None)
    message = " ".join(str(exc).split())
    if status:
        message = f"HTTP {status}: {message}"
    return message[:limit] or type(exc).__name__


def _is_quota_exhausted(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "resource_exhausted" in message
        or "quota exceeded" in message
        or "per day limit exceeded" in message
        or "generate_content_free_tier_requests" in message
    )


def _is_transient(exc: Exception) -> bool:
    # A 400/BadRequest is a permanent request/configuration error: retrying it
    # cannot help. Retry only known temporary service/network failures.
    if getattr(exc, "status_code", None) in (400, 401, 403, 404, 422):
        return False
    message = str(exc).lower()
    return any(code in message for code in (
        "429", "500", "502", "503", "504", "rate_limit",
        "overloaded", "temporarily", "timeout", "connection"
    ))


def invoke_with_resilience(prompt: str) -> str:
    """Use Groq first, then Gemini; log actionable provider errors without secrets."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    primary_error = None

    if groq_key:
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=180.0,
            max_retries=0,
        )
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.responses.create(
                    model=GROQ_MODEL,
                    input=prompt,
                    reasoning={"effort": "low"},
                    store=False,
                )
                generated = _extract_text(response)
                if not generated:
                    raise RuntimeError("Groq returned an empty response.")
                return generated
            except Exception as exc:
                primary_error = exc
                logger.warning(
                    "Groq request failed (model=%s, attempt=%s/%s): %s",
                    GROQ_MODEL, attempt + 1, MAX_RETRIES + 1,
                    _error_summary(exc),
                )
                if not _is_transient(exc) or attempt >= MAX_RETRIES:
                    break
                time.sleep(1.0 * (2 ** attempt))
    else:
        primary_error = RuntimeError("GROQ_API_KEY is missing.")
        logger.error("Groq is not configured: GROQ_API_KEY is missing.")

    logger.info("Attempting Gemini fallback after Groq failure.")
    try:
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
        fallback_summary = _error_summary(fallback_error)
        logger.error("Gemini fallback failed: %s", fallback_summary)

        if _is_quota_exhausted(fallback_error):
            raise RuntimeError(
                "Groq generation failed: "
                + _error_summary(primary_error, 450)
                + ". Gemini fallback is currently unavailable because its API quota "
                "has been exhausted. Check Gemini usage/billing or wait for quota reset. "
                "See https://ai.dev/rate-limit."
            ) from fallback_error

        raise RuntimeError(
            "Groq generation failed: "
            + _error_summary(primary_error, 450)
            + ". Gemini fallback also failed: "
            + fallback_summary
        ) from fallback_error


def parse_json_array(raw: str):
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I)
    start = cleaned.find("[")
    if start < 0:
        raise ValueError("The AI response did not contain a JSON array.")
    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("The AI response contained invalid JSON.") from exc
    if not isinstance(data, list):
        raise ValueError("The AI response had an unexpected format.")
    return data


def invoke_json_array(prompt: str, stage_name: str):
    """Call a provider and make one format-repair attempt if its JSON is malformed."""
    raw = invoke_with_resilience(prompt)
    try:
        return parse_json_array(raw)
    except ValueError:
        logger.warning("%s returned malformed JSON; requesting one format correction.", stage_name)
        repair_prompt = (
            "Convert the following model output into a valid JSON array. "
            "Preserve its information without adding new facts. Return only JSON, "
            "with no markdown fences or commentary.\n\n"
            f"MODEL OUTPUT TO REPAIR:\n{json.dumps(raw[:18000], ensure_ascii=False)}"
        )
        return parse_json_array(invoke_with_resilience(repair_prompt))


def generate_interview_questions(chunks, topic: str, role: str, difficulty: str, count: int):
    """Stage 1: generate questions from the complete retrieved context."""
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'document')} | chunk {doc.metadata.get('chunk_id', '?')}]\n"
        f"{doc.page_content}"
        for doc in chunks
    )
    question_prompt = PromptTemplate.from_template(
        """You are an expert technical interviewer.
Generate exactly {count} distinct interview questions.
Topic/focus: {topic}
Target role: {role}
Difficulty: {difficulty}

Use the retrieved context as the factual basis. Questions should test understanding and
practical application, fit the target role, and avoid duplicates. Do not include answers.
Return ONLY a valid JSON array. Each object must contain "question", "concept", "source_hint".
No markdown fences or commentary.

RETRIEVED CONTEXT:
{context}"""
    ).format(
        count=count,
        topic=topic or "Relevant topics from uploaded material",
        role=role or "General technical role",
        difficulty=difficulty,
        context=context,
    )
    raw_questions = invoke_json_array(question_prompt, "Question generation")
    questions = []
    for item in raw_questions:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        questions.append({
            "question_id": f"Q{len(questions) + 1}",
            "question": str(item["question"]).strip(),
            "concept": str(item.get("concept", "")).strip(),
            "source_hint": str(item.get("source_hint", "")).strip(),
        })
        if len(questions) >= count:
            break
    if len(questions) != count:
        raise ValueError(
            f"Question generation returned {len(questions)} of {count} requested questions. Please retry."
        )
    return questions


class RefineAnswerChain:
    """Sequential Refine chain: seed answers from one document, then refine with each next document.

    Each model call handles the full question batch, keeping the number of calls proportional
    to retrieved documents rather than multiplying it by the number of questions.
    """

    initial_prompt = PromptTemplate.from_template(
        """You are an expert technical interview coach.
Create an accurate, interview-ready initial answer for every question.
Target role: {role}
Difficulty: {difficulty}

Use ONLY the source passage below for factual claims. Explain clearly and include a concise
example only when supported by the passage. If the passage does not contain enough evidence,
say what is missing rather than inventing details.
Return ONLY a valid JSON array with one object per question, using exactly "question_id" and "answer".
Preserve every question_id. No markdown fences or commentary.

QUESTIONS:
{questions}

INITIAL SOURCE PASSAGE ({source_label}):
{context}"""
    )

    refine_prompt = PromptTemplate.from_template(
        """You are refining draft interview answers using additional retrieved evidence.
Target role: {role}
Difficulty: {difficulty}

For every question, improve the existing answer using the NEW SOURCE PASSAGE.
- Preserve correct information already in the draft.
- Add useful details, definitions, examples, or qualifications supported by the new passage.
- Resolve contradictions cautiously; do not invent facts or claim unsupported certainty.
- If the passage is irrelevant to a question, leave that answer substantively unchanged.
- Keep answers concise, clear, and suitable to speak in an interview.
Return ONLY a valid JSON array containing every question_id and its refined "answer".
Do not omit any question_id. No markdown fences or commentary.

QUESTIONS:
{questions}

CURRENT DRAFT ANSWERS:
{current_answers}

NEW SOURCE PASSAGE ({source_label}):
{context}"""
    )

    def __init__(self, role: str, difficulty: str, progress_callback=None):
        self.role = role or "General technical role"
        self.difficulty = difficulty
        self.progress_callback = progress_callback

    def invoke(self, questions, documents):
        if not documents:
            raise ValueError("Refine chain requires at least one retrieved source passage.")

        # Initial answer pass over the first retrieved document.
        first = documents[0]
        if self.progress_callback:
            self.progress_callback(
                f"Stage 2/2 · Initial answers from source chunk 1/{len(documents)}"
            )
        initial_prompt = self.initial_prompt.format(
            role=self.role,
            difficulty=self.difficulty,
            questions=json.dumps(questions, ensure_ascii=False),
            source_label=f"{first.metadata.get('source', 'document')} · chunk {first.metadata.get('chunk_id', '?')}",
            context=first.page_content,
        )
        initial_items = invoke_json_array(initial_prompt, "Initial answer generation")
        answers = self._normalise_items(initial_items, questions)
        answer_ids = {item["question_id"] for item in answers}
        expected_ids = {item["question_id"] for item in questions}
        if answer_ids != expected_ids:
            missing_ids = sorted(expected_ids - answer_ids)
            raise ValueError(
                "The initial answer pass missed question(s): "
                + ", ".join(missing_ids)
                + ". Please retry generation."
            )

        # Refine the complete answer batch sequentially with each remaining retrieved document.
        for index, document in enumerate(documents[1:], start=2):
            if self.progress_callback:
                self.progress_callback(
                    f"Refine chain · incorporating source chunk {index}/{len(documents)}"
                )
            refine_prompt = self.refine_prompt.format(
                role=self.role,
                difficulty=self.difficulty,
                questions=json.dumps(questions, ensure_ascii=False),
                current_answers=json.dumps(answers, ensure_ascii=False),
                source_label=f"{document.metadata.get('source', 'document')} · chunk {document.metadata.get('chunk_id', '?')}",
                context=document.page_content,
            )
            refined_items = invoke_json_array(refine_prompt, "Answer refinement")
            refined = self._normalise_items(refined_items, questions)

            # Keep the previous answer if a provider omits one ID in a refinement response.
            refined_by_id = {item["question_id"]: item["answer"] for item in refined}
            answers = [
                {
                    "question_id": current["question_id"],
                    "answer": refined_by_id.get(current["question_id"], current["answer"]),
                }
                for current in answers
            ]

        return answers

    @staticmethod
    def _normalise_items(items, questions):
        allowed_ids = {question["question_id"] for question in questions}
        normalised = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("question_id", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question_id in allowed_ids and answer:
                normalised[question_id] = {
                    "question_id": question_id,
                    "answer": answer,
                }
        return list(normalised.values())


def generate_interview_set(chunks, topic: str, role: str, difficulty: str, count: int, progress_callback=None):
    """Two-stage pipeline: question generation followed by a sequential Refine answer chain."""
    if not chunks:
        raise ValueError("No relevant source passages were retrieved. Try another topic or document.")

    if progress_callback:
        progress_callback("Stage 1/2 · Generating interview questions")
    questions = generate_interview_questions(chunks, topic, role, difficulty, count)

    refine_chain = RefineAnswerChain(
        role=role,
        difficulty=difficulty,
        progress_callback=progress_callback,
    )
    refined_answers = refine_chain.invoke(questions, chunks)
    answer_by_id = {
        item["question_id"]: item["answer"]
        for item in refined_answers
    }

    results = []
    for question in questions:
        answer = answer_by_id.get(question["question_id"])
        if not answer:
            raise ValueError(
                f"No refined answer was produced for {question['question_id']}. Please retry."
            )
        results.append({
            **question,
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
    return results


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
            chunks = None
            vector_store = None
            query_embeddings = None
            retrieved = None
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

                    # Preserve only the content required by the results UI.
                    saved_chunks = [
                        {
                            "source": chunk.metadata.get("source", "document"),
                            "source_index": chunk.metadata.get("source_index", 0),
                            "chunk_id": chunk.metadata.get("chunk_id", 0),
                            "text": chunk.page_content,
                        }
                        for chunk in chunks
                    ]
                    saved_retrieved_keys = {
                        (chunk.metadata.get("source", "document"), chunk.metadata.get("chunk_id", 0))
                        for chunk in retrieved
                    }
                    saved_run_meta = {
                        "files": [f.name for f in files],
                        "chunks": len(chunks),
                        "retrieved": len(retrieved),
                        "topic": topic or "Document topics",
                        "role": role or "General technical role",
                        "difficulty": difficulty,
                    }
                    status.update(label="Your interview set is ready", state="complete", expanded=False)

                st.session_state["results"] = results
                st.session_state["all_chunks"] = saved_chunks
                st.session_state["retrieved_chunk_keys"] = saved_retrieved_keys
                st.session_state["run_meta"] = saved_run_meta
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
            finally:
                # Drop references to temporary RAG objects on both success and failure.
                retrieved = None
                vector_store = None
                query_embeddings = None
                chunks = None

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
                if item.get("sources"):
                    with st.expander(f"Sources used · {len(item['sources'])} retrieved passages"):
                        for source in item["sources"]:
                            st.markdown(
                                f"**{source['source']} · Chunk {source['chunk_id']}**"
                            )
                            st.caption(source["text"])
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
        '<div class="small-note"><strong>How it works:</strong> Document extraction → Recursive chunking → Gemini embeddings → FAISS semantic search → Question generation → Initial answer → Sequential Refine chain → Interview-ready Q&A</div>'
        '<div class="small-note">Gemini creates embeddings · Groq generates questions and runs the Refine chain, with Gemini fallback · Documents and FAISS index are processed in memory for this session.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
