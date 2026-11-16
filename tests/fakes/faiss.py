import pickle, numpy as np
class IndexFlatIP:
    def __init__(self, d): self.d = d
class IndexIDMap:
    def __init__(self, base): self.d = base.d; self.v = np.zeros((0, base.d), "float32"); self.ids = np.zeros(0, "int64")
    def add_with_ids(self, x, ids): self.v = np.vstack([self.v, x]); self.ids = np.concatenate([self.ids, ids])
    def search(self, q, k):
        s = q @ self.v.T; o = np.argsort(-s, axis=1)[:, :k]
        sc = np.take_along_axis(s, o, 1); ids = self.ids[o]
        if ids.shape[1] < k:
            pad = k - ids.shape[1]; sc = np.pad(sc, ((0,0),(0,pad))); ids = np.pad(ids, ((0,0),(0,pad)), constant_values=-1)
        return sc, ids
def write_index(i, p): pickle.dump(i, open(p, "wb"))
def read_index(p): return pickle.load(open(p, "rb"))
