"""
All settings live here. The API key is set ONLY in this file (or via the
GROQ_API_KEY environment variable). It is never shown in the Streamlit UI.
"""
import os

# >>> PASTE YOUR GROQ API KEY HERE <<<
GROQ_API_KEY = os.getenv("GROQ_API_KEY","PASTE YOUR API")

GROQ_MODEL = "openai/gpt-oss-120b"      # any Groq chat model id
TEMPERATURE = 0.1                           # low = more factual
MAX_OUTPUT_TOKENS = 1200

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # runs locally, free
EMBED_BATCH_SIZE = 64

CHUNK_SIZE = 900        # characters
CHUNK_OVERLAP = 150

TOP_K = 12              # chunks sent to the LLM
MAX_ROSTER_NAMES = 300  # candidate names listed to the LLM (for count/list questions)
HISTORY_TURNS = 4       # previous Q&A turns remembered

INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "index")
