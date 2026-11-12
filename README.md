# Prepwise — Interview Question Generator

A document-grounded interview preparation application built with Streamlit, LangChain, FAISS, Gemini embeddings, Grok, and Gemini fallback.

## Features
- Upload PDF, DOCX, TXT, or Markdown source material.
- Extract and clean document text, then split it into overlapping chunks.
- Generate text embeddings with Google's `gemini-embedding-001` model.
- Store chunk vectors in an in-memory FAISS index and retrieve the top 5 semantically relevant passages.
- Generate interview questions and answers in two distinct LLM stages.
- Grok is the primary generation provider; Gemini is the automatic fallback.
- Display source chunks in document order and export interview sets as Markdown.
- Provider credentials are configured server-side and never shown in the UI.

## RAG and generation pipeline

```text
Document upload (PDF / DOCX / TXT / MD)
        ↓
Text extraction and cleaning
        ↓
Recursive chunking (700 characters, 80 overlap)
        ↓
Gemini Embeddings (gemini-embedding-001)
        ↓
FAISS vector index (in-memory)
        ↓
Semantic similarity search (top 5 chunks)
        ↓
Prompt 1 — Generate interview questions
        ↓
Prompt 2 — Generate answers for those questions
        ↓
Grok primary · Gemini fallback for each generation stage
        ↓
Interview-ready Q&A with source context
```

FAISS is the vector index. Gemini generates the embeddings. The index is created in memory for each generation run and is not persisted to disk or a hosted vector database.

## Render environment variables

| Variable | Value |
|---|---|
| `XAI_API_KEY` | Your xAI API key |
| `GROK_MODEL` | `grok-4.7` |
| `GEMINI_API_KEY` | Google Gemini API key (required for embeddings and fallback) |
| `GEMINI_MODEL` | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | `gemini-embedding-001` |

Save environment variables and redeploy the Render service.

## Local setup
1. Use Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Add the environment variables above to `.env`.
5. Run: `streamlit run app.py`.

## Performance and reliability
- FAISS performs vector similarity search over document embeddings.
- Question generation and answer generation are two separate batched LLM calls, not one call per answer.
- Grok requests retry transient failures; Gemini is attempted if Grok fails.
- The FAISS index and document chunks are held in memory for the active generation run.
- Uploads are limited to 20 MB per file; extracted text is capped at 250,000 characters and chunk processing at 300 chunks.

Embedding generation and both LLM stages require provider connectivity. If embedding or both generation providers fail, check Render logs and verify API keys, model access, and quota.

## Privacy
There is no login system or persistent document/vector database. Uploaded files, extracted chunks, and the FAISS index are processed in memory for the active session/run. Document text is sent to Google's embedding API to create vectors. Retrieved text is sent to Grok, or Gemini if generation fallback is used. Do not upload confidential material unless you are comfortable sharing it with those providers.

Generated answers may contain mistakes; verify them before relying on them in an interview.
