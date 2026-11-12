# Prepwise — Interview Question Generator

A document-grounded interview preparation app built with Streamlit, LangChain text splitting, lightweight lexical retrieval, and Gemini. Upload your own learning material and generate role-focused interview questions with suggested answers.

## What it does
- Accepts PDF, DOCX, TXT, and Markdown source material.
- Extracts and cleans text, then creates overlapping chunks.
- Uses lightweight local keyword ranking to retrieve relevant chunks (no embedding model or vector database required).
- Generates questions and answers grounded in the retrieved passages.
- Shows source passages and supports Markdown export.
- Keeps provider credentials and model configuration on the server; they are never entered in the UI.

## RAG pipeline

```text
Upload documents
      ↓
Text extraction (PDF / DOCX / TXT / MD)
      ↓
Text cleaning
      ↓
Recursive chunking (700 chars, 80 overlap)
      ↓
Lightweight lexical retrieval (top 5 chunks)
      ↓
Gemini generates questions + answers in one request
      ↓
Source-backed results + Markdown export
```

The retrieval architecture remains deliberately lightweight for Render's small instances. The latency improvement comes from combining question and answer generation into a single model request instead of making one request for questions followed by a separate sequential request for every answer.

## Local setup
1. Use Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and set your Gemini API key.
5. Run: `streamlit run app.py`.

## Render deployment
Connect this repository to Render as a Python web service. The included `render.yaml` contains the build and start commands.

In **Render → your service → Environment**, set:
- `GEMINI_API_KEY` — required secret from Google AI Studio.
- `GEMINI_MODEL` — optional; defaults to `gemini-2.5-flash`.
- `GEMINI_FALLBACK_MODEL` — optional fallback model name from your Gemini project, used only if the primary model encounters a temporary capacity/rate-limit error.

Save the environment variables and trigger a redeploy. The API key and model are intentionally not exposed in the app interface.

## Reliability and performance
- Transient Gemini 429/500/503 errors receive bounded exponential-backoff retries.
- An optional fallback model can be configured for provider capacity issues.
- Question and answer generation happen in one model round-trip to reduce latency.
- Generation requests use a bounded timeout.
- Uploads are limited to 20 MB per file; extracted text is capped at 250,000 characters and processing is capped at 300 chunks.

A provider-side outage or exhausted quota cannot be eliminated by application code. If the provider remains unavailable after retries and fallback, the app shows a concise actionable message rather than a raw stack trace.

## Privacy
There is no login system or persistent document database. Uploaded files are processed for the active Streamlit session. Retrieved text is sent to the configured Gemini API as part of the generation prompt. Do not upload confidential material unless you are comfortable sharing it with the provider.

Generated answers may contain mistakes; verify them before relying on them in an interview.
