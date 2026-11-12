# Prepwise — Interview Question Generator

A document-grounded interview preparation application built with Streamlit, LangChain, FAISS, Gemini embeddings, Groq, and Gemini fallback.

## Implemented architecture

Document upload (PDF / DOCX / TXT / Markdown)
→ Text extraction and cleaning
→ RecursiveCharacterTextSplitter (700-character chunks, 80-character overlap)
→ Gemini Embeddings (gemini-embedding-001)
→ FAISS in-memory vector index
→ Semantic similarity retrieval (top 5 passages)
→ Stage 1: question-generation prompt
→ Stage 2: answer generation with a sequential Refine chain
→ Initial answer from first retrieved passage
→ Sequential refinement using each remaining passage
→ Final interview Q&A with source attribution

## Retrieval and embeddings
- Chunking: LangChain RecursiveCharacterTextSplitter, 700 characters with 80-character overlap.
- Embedding model: Google gemini-embedding-001.
- Vector index: FAISS via LangChain Community FAISS.
- Document embeddings use the RETRIEVAL_DOCUMENT task type.
- Query embeddings use the RETRIEVAL_QUERY task type.
- Retrieval: FAISS vector similarity search; top 5 passages are selected and restored to source-document order before the Refine chain.

FAISS is the vector store actually used by the application. ChromaDB is not currently part of the runtime architecture; do not list ChromaDB as an implemented technology on the resume unless it is added separately.

## Two-stage generation and Refine chain
1. Question generation: one prompt generates the requested set of role- and difficulty-specific questions from all retrieved passages.
2. Answer generation: a custom sequential LangChain-prompt Refine chain generates an initial batch of answers from the first retrieved passage, then refines that complete answer batch with each subsequent passage. Existing supported information is retained and new supported evidence is incorporated. Each answer is matched using a stable question ID.

The question set is generated in one model call. The answer Refine chain makes one initial call plus one refinement call per additional retrieved passage. With five retrieved passages, this is one question-generation call and up to five answer-generation/refinement calls. Batching all questions in each pass avoids making a separate call for every question.

## Model routing
- Primary generation provider: GroqCloud through the Groq Responses API.
- Generation fallback: Gemini (gemini-2.5-flash) is attempted when Grok fails.
- Embedding provider: Gemini Embeddings. This is used for document and query vectors even when Grok handles text generation.
- Transient Groq failures receive bounded retries. If Groq still fails, the same generation step falls back to Gemini.
- Malformed JSON receives one format-correction attempt.

## Render environment variables

Set these in Render → Service → Environment:

| Variable | Value |
|---|---|
| GROQ_API_KEY | Your GroqCloud API key (starts with `gsk_`) |
| GROQ_MODEL | openai/gpt-oss-120b |
| GEMINI_API_KEY | Google Gemini API key (required for embeddings and fallback) |
| GEMINI_MODEL | gemini-2.5-flash |
| EMBEDDING_MODEL | gemini-embedding-001 |

Save environment variables and redeploy the Render service. API keys are never requested in the browser.

## Local setup
1. Use Python 3.11 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: pip install -r requirements.txt.
4. Add the environment variables above to .env.
5. Run: streamlit run app.py.

## Runtime and privacy
- Uploaded files, extracted chunks, and the FAISS index are held in memory for the active generation run; no persistent vector database is configured.
- Document text is sent to Google's embedding API to create vectors.
- Retrieved text is sent to Groq, or Gemini when generation fallback is used.
- Uploads are limited to 20 MB per file; extracted text is capped at 250,000 characters and chunk processing at 300 chunks.
- The Refine chain improves coverage across retrieved passages but adds sequential model calls; actual latency depends on provider response times and quota.
- Generated answers may contain mistakes; verify them before relying on them in an interview.
