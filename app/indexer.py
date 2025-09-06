# app/indexer.py
import faiss
import numpy as np
import json
import os

class FaissIndexer:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.docs = []
        self.meta = []

    def add(self, vectors: np.ndarray, texts: list, metas: list = None):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self.index.add(vectors.astype('float32'))
        self.docs.extend(texts)
        if metas:
            self.meta.extend(metas)
        else:
            for _ in texts:
                self.meta.append({})

    def search(self, vector: np.ndarray, top_k: int = 5):
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        D, I = self.index.search(vector.astype('float32'), top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append({
                    "text": self.docs[idx],
                    "meta": self.meta[idx]
                })
        return results

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load(self, folder: str):
        self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
