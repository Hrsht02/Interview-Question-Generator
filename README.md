# LLM-Based Interview Question Generator

A document-grounded interview question and answer generator built with Python, Streamlit, LangChain, Gemini API, Hugging Face embeddings, and ChromaDB. No login system is included.

## Features
- Upload PDF, DOCX, TXT, or Markdown source material
- Text extraction, cleaning, chunking, and embedding generation
- Local Chroma vector store for semantic retrieval
- Gemini-powered retrieval-augmented question generation
- Two-stage generation: create questions, then draft answers grounded in retrieved context
- Iterative refine step across retrieved chunks
- Adjustable question count, difficulty, and topic
- Source chunk references and download of generated results

## Setup
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` from Google AI Studio.
5. Run: `streamlit run app.py`

## Notes
- Uploaded files are processed by the app; the vector index is held in the current session and is not a hosted database.
- Keep your API key private. Never commit `.env`.
- Model availability and quotas depend on your Gemini API project. Set `GEMINI_MODEL` to a model enabled for your key.
- Generated content can be inaccurate; review answers before relying on them.

## Pipeline
Documents → extraction/cleaning → overlapping chunks → Hugging Face embeddings → Chroma similarity search → Gemini question generation/refinement → separate Gemini answer generation → results with source excerpts.
