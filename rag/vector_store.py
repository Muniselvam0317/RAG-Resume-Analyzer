"""FAISS vector store with metadata and on-disk persistence."""
import json
import os
from typing import Dict, List, Tuple

import faiss
import numpy as np


class ResumeVectorStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "records.json")
        self.index = None
        self.records: Dict[int, dict] = {}
        self.next_id = 0
        os.makedirs(index_dir, exist_ok=True)
        self._load()

    # ---------- persistence ----------
    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.records = {int(k): v for k, v in meta["records"].items()}
            self.next_id = meta["next_id"]

    def save(self):
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"records": self.records, "next_id": self.next_id}, f)

    # ---------- queries ----------
    @property
    def size(self) -> int:
        return len(self.records)

    def resume_ids(self) -> set:
        return {r["resume_id"] for r in self.records.values()}

    def candidates(self) -> List[Tuple[str, str]]:
        """Unique (candidate, file) pairs."""
        seen = {}
        for r in self.records.values():
            seen[r["resume_id"]] = (r["candidate"], r["file"])
        return sorted(seen.values())

    # ---------- mutations ----------
    def add(self, vectors: np.ndarray, records: List[dict]):
        if len(records) == 0:
            return
        if self.index is None:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
        ids = np.arange(self.next_id, self.next_id + len(records), dtype="int64")
        self.index.add_with_ids(vectors.astype("float32"), ids)
        for i, rec in zip(ids, records):
            self.records[int(i)] = rec
        self.next_id += len(records)
        self.save()

    def clear(self):
        self.index, self.records, self.next_id = None, {}, 0
        for p in (self.index_path, self.meta_path):
            if os.path.exists(p):
                os.remove(p)

    # ---------- search ----------
    def search(self, qvec: np.ndarray, k: int) -> List[Tuple[dict, float]]:
        if self.index is None or self.size == 0:
            return []
        k = min(k, self.size)
        scores, ids = self.index.search(qvec.astype("float32"), k)
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i != -1 and int(i) in self.records:
                out.append((self.records[int(i)], float(s)))
        return out
