"""Offline smoke test: real loader/chunker/store/pipeline; stub embedder + stub LLM."""
import io, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import faiss  # noqa
    print("using REAL faiss")
except ImportError:
    sys.path.insert(0, os.path.join(ROOT, "tests", "fakes")); print("using FAKE faiss")

import numpy as np, docx
from sklearn.feature_extraction.text import HashingVectorizer
from rag.pipeline import ResumeRAG
from rag.vector_store import ResumeVectorStore

class StubEmbedder:
    hv = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2")
    def encode(self, texts): return self.hv.transform(texts).toarray().astype("float32")

class StubLLM:
    def stream(self, messages):
        self.last = messages
        yield "stub answer"

def make_docx(lines):
    d = docx.Document()
    for l in lines: d.add_paragraph(l)
    b = io.BytesIO(); d.save(b); return b.getvalue()

resumes = {
 "alice.docx": make_docx(["Alice Johnson", "alice@x.com", "Skills: Python, AWS, Docker, Kubernetes",
                          "Experience: 6 years as Backend Engineer at Acme building Python microservices on AWS."]),
 "bob.txt": "Bob Smith\nSkills: Java, Spring, Oracle\nExperience: 4 years at Globex as Java developer in banking.".encode(),
 "carol.txt": "Carol White\nSkills: Tableau, SQL, Excel\nExperience: 3 years data analyst, dashboards and reporting.".encode(),
 "bad.xyz": b"nope",
 "empty.txt": b"hi",
}
tmp = tempfile.mkdtemp()
rag = ResumeRAG(embedder=StubEmbedder(), llm=StubLLM(), store=ResumeVectorStore(tmp))
res = rag.ingest(list(resumes.items()))
print(res)
assert len(res["added"]) == 3 and len(res["failed"]) == 2
res2 = rag.ingest([("alice.docx", resumes["alice.docx"])]); assert res2["skipped"]
assert [c for c, _ in rag.store.candidates()] == ["Alice Johnson", "Bob Smith", "Carol White"]

hits = rag.retrieve("Who knows Python and AWS?", 3)
assert hits[0][0]["candidate"] == "Alice Johnson", hits[0][0]
hits = rag.retrieve("Summarize Bob Smith", 5)
assert {h[0]["candidate"] for h in hits} == {"Bob Smith"}

stream, sources = rag.ask("Who knows Python?", [])
assert "".join(stream) == "stub answer" and sources
user_msg = rag.llm.last[-1]["content"]
assert "CANDIDATE ROSTER (3 resumes" in user_msg and "Alice Johnson" in user_msg

# persistence: reload from disk
rag2 = ResumeRAG(embedder=StubEmbedder(), llm=StubLLM(), store=ResumeVectorStore(tmp))
assert rag2.store.size == rag.store.size
rag2.store.clear(); assert rag2.store.size == 0
assert "upload" in "".join(rag2.ask("hi")[0]).lower()
print("ALL OFFLINE TESTS PASSED")
