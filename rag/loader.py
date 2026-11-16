"""Extract text from PDF / DOCX / TXT resumes."""
import io
import os
import re

from pypdf import PdfReader
import docx

SUPPORTED = (".pdf", ".docx", ".txt")


def extract_text(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext == ".docx":
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:                      # resumes often use tables
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        text = "\n".join(parts)
    elif ext == ".txt":
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return clean_text(text)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_candidate_name(filename: str, text: str) -> str:
    """Use the first line if it looks like a name, else the file name."""
    for line in text.splitlines()[:5]:
        line = line.strip()
        words = line.split()
        if (2 <= len(words) <= 4 and not re.search(r"[\d@:/|]", line)
                and all(w[0].isalpha() for w in words)):
            return line.title() if line.isupper() else line
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[_\-]+", " ", stem).strip()
