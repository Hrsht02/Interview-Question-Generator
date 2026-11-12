# Prepwise — Interview Question Generator

A document-grounded interview preparation app built with Streamlit, LangChain text splitting, lightweight lexical retrieval, Grok, and Gemini fallback.

## Features
- Upload PDF, DOCX, TXT, or Markdown source material.
- Text extraction, cleaning, and overlapping chunking.
- Lightweight local keyword retrieval (no embedding model or vector database).
- Grok-powered interview question and answer generation.
- Automatic Gemini fallback if Grok generation fails.
- Source passages and Markdown export.
- Provider credentials are configured server-side and never shown in the UI.

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
Grok generates questions + answers in one request
      ↓
Gemini fallback if Grok fails
      ↓
Source-backed results + Markdown export
```

The retrieval architecture is unchanged. Latency is reduced by generating questions and answers in one model request instead of making separate sequential requests for every answer.

## Render environment variables

Set these in **Render → Service → Environment**:

| Variable | Value |
|---|---|
| `XAI_API_KEY` | Your xAI API key |
| `GROK_MODEL` | `grok-4.7` |
| `GEMINI_API_KEY` | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` |

The app calls the xAI Responses API at `https://api.x.ai/v1`. Gemini is attempted automatically if Grok fails. Both keys must be valid and have available quota for fallback to work.

After changing environment variables, save and redeploy the Render service.

## Local setup
1. Use Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Add the four environment variables above to `.env`.
5. Run: `streamlit run app.py`.

## Reliability and performance
- Grok primary requests use bounded retries for transient provider failures.
- Gemini is attempted as a fallback after Grok fails.
- Questions and answers are generated in one model request.
- API keys are never accepted from browser inputs.
- Uploads are limited to 20 MB per file; extracted text is capped at 250,000 characters and processing at 300 chunks.

If both providers are unavailable, check Render logs for server-side diagnostic details.

## Privacy
There is no login system or persistent document database. Uploaded files are processed for the active Streamlit session. Retrieved text is sent to Grok, or Gemini if fallback is used. Do not upload confidential material unless you are comfortable sharing it with the providers.

Generated answers may contain mistakes; verify them before relying on them in an interview.
