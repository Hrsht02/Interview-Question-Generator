# LLM-Based Interview Question Generator

A document-grounded interview question and answer generator built with Python, Streamlit, LangChain's text splitter, and the Gemini API. No login system is included.

## Features
- Upload PDF, DOCX, TXT, or Markdown source material
- Text extraction, cleaning, and overlapping chunking
- Lightweight local keyword-based retrieval (no local neural embedding model)
- Gemini-powered interview question generation and answer drafting
- Adjustable question count, difficulty, topic, and target role
- Source chunk references and Markdown download

## Setup
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` from Google AI Studio.
5. Run locally: `streamlit run app.py`

## Render deployment
The repository includes `render.yaml`. Connect the repository to Render as a web service and set `GEMINI_API_KEY` in the service's environment variables. The start command must be:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

The app deliberately avoids loading local sentence-transformer models and creating an in-memory Chroma index, which can exceed the 512 MB memory limit on small instances. Retrieval is lightweight and local; Gemini is used for generation. Large document sets are capped at 300 chunks per run to limit memory use.

## Storage and privacy
No SQL database is required for the current design: there are no user accounts and uploaded files/results are session-based rather than persistent. The app does not intentionally store uploaded documents in a database. Relevant document text is sent to Gemini as part of generation prompts, so do not upload confidential material unless you are comfortable sharing it with the API provider.

## Environment variables
- `GEMINI_API_KEY` — required for generation
- `GEMINI_MODEL` — optional; must be available to your Gemini API project (default: `gemini-2.5-flash`)

## Pipeline
Upload → text extraction → chunking → lightweight keyword ranking → Gemini question generation → Gemini answer generation → display and Markdown export.

Generated content can be inaccurate; review answers before relying on them.
