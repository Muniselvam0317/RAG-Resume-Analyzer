"""Ingestion + retrieval + generation, wired together."""
import hashlib
from typing import Dict, Iterator, List, Optional, Tuple

import config
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.loader import SUPPORTED, extract_text, guess_candidate_name
from rag.vector_store import ResumeVectorStore
from rag.llm import SYSTEM_PROMPT


class ResumeRAG:
    def __init__(self, embedder=None, llm=None, store: Optional[ResumeVectorStore] = None):
        self.embedder = embedder or Embedder(config.EMBEDDING_MODEL, config.EMBED_BATCH_SIZE)
        self.store = store or ResumeVectorStore(config.INDEX_DIR)
        self._llm = llm  # created lazily so the UI can load without a key

    @property
    def llm(self):
        if self._llm is None:
            from rag.llm import GroqLLM
            self._llm = GroqLLM(config.GROQ_API_KEY, config.GROQ_MODEL,
                                config.TEMPERATURE, config.MAX_OUTPUT_TOKENS)
        return self._llm

    # ------------------------------------------------------------ ingestion
    def ingest(self, files: List[Tuple[str, bytes]], progress=None) -> Dict[str, list]:
        """files: [(filename, bytes)] -> {'added': [], 'skipped': [], 'failed': []}"""
        result = {"added": [], "skipped": [], "failed": []}
        known = self.store.resume_ids()
        total = len(files)
        for n, (name, data) in enumerate(files, 1):
            try:
                if not name.lower().endswith(SUPPORTED):
                    raise ValueError("unsupported file type")
                rid = hashlib.sha256(data).hexdigest()[:16]
                if rid in known:
                    result["skipped"].append(f"{name} (already indexed)")
                    continue
                text = extract_text(name, data)
                if len(text) < 30:
                    raise ValueError("no extractable text (scanned image PDF?)")
                candidate = guess_candidate_name(name, text)
                chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                # prefix each chunk with the candidate so embeddings keep identity
                embed_inputs = [f"Candidate: {candidate}\n{c}" for c in chunks]
                vecs = self.embedder.encode(embed_inputs)
                records = [
                    {"resume_id": rid, "file": name, "candidate": candidate,
                     "chunk_index": i, "text": c}
                    for i, c in enumerate(chunks)
                ]
                self.store.add(vecs, records)
                known.add(rid)
                result["added"].append(f"{name} → {candidate} ({len(chunks)} chunks)")
            except Exception as e:  # keep going on bad files
                result["failed"].append(f"{name}: {e}")
            finally:
                if progress:
                    progress(n / total, name)
        return result

    # ------------------------------------------------------------ retrieval
    def _mentioned(self, question: str) -> set:
        q = question.lower()
        return {c for c, _ in self.store.candidates() if len(c) > 3 and c.lower() in q}

    def retrieve(self, question: str, top_k: int = None) -> List[Tuple[dict, float]]:
        top_k = top_k or config.TOP_K
        if self.store.size == 0:
            return []
        qvec = self.embedder.encode([question])
        mentioned = self._mentioned(question)
        if mentioned:  # question is about a named person -> restrict to them
            hits = self.store.search(qvec, self.store.size)
            hits = [h for h in hits if h[0]["candidate"] in mentioned]
            return hits[:top_k]
        return self.store.search(qvec, top_k)

    # ------------------------------------------------------------ generation
    def _build_messages(self, question, hits, history) -> List[Dict[str, str]]:
        cands = self.store.candidates()
        roster = "\n".join(f"- {c} ({f})" for c, f in cands[: config.MAX_ROSTER_NAMES])
        if len(cands) > config.MAX_ROSTER_NAMES:
            roster += f"\n... and {len(cands) - config.MAX_ROSTER_NAMES} more"
        context = "\n\n".join(
            f"[Excerpt {i}] Candidate: {r['candidate']} | File: {r['file']}\n{r['text']}"
            for i, (r, _) in enumerate(hits, 1)
        )
        user = (f"CANDIDATE ROSTER ({len(cands)} resumes indexed):\n{roster}\n\n"
                f"RESUME EXCERPTS:\n{context}\n\nQUESTION: {question}")
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in (history or [])[-2 * config.HISTORY_TURNS:]:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user})
        return msgs

    def ask(self, question: str, history: List[Dict[str, str]] = None
            ) -> Tuple[Iterator[str], List[dict]]:
        if self.store.size == 0:
            return iter(["Please upload and process some resumes first."]), []
        hits = self.retrieve(question)
        sources = [{**r, "score": round(s, 3)} for r, s in hits]
        return self.llm.stream(self._build_messages(question, hits, history)), sources
