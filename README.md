# Resume Analyzer (RAG) — Streamlit + FAISS + Groq

## Setup
1. `python -m venv venv && source venv/bin/activate`   (Windows: `venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Open `config.py` and paste your key into `GROQ_API_KEY` (the key is never shown in the UI).
4. `streamlit run app.py`

## Use
Upload any number of PDF/DOCX/TXT resumes in the sidebar → **Process resumes** → chat.
The FAISS index is saved in `data/index/`, so resumes persist between runs.
Duplicate files are skipped automatically. Scanned (image-only) PDFs need OCR first.
