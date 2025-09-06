# app/qa.py
from transformers import pipeline

class RetrievalQA:
    def __init__(self, embedder, indexer, qa_model_name: str = "deepset/roberta-base-squad2", device: int = -1):
        self.embedder = embedder
        self.indexer = indexer
        self.qa = pipeline("question-answering", model=qa_model_name, device=device)

    def answer(self, question: str, top_k: int = 5):
        qvec = self.embedder.embed(question)
        hits = self.indexer.search(qvec, top_k=top_k)
        context = "\n\n".join([h["text"] for h in hits])
        if not context.strip():
            return {"answer": "", "source_chunks": []}
        res = self.qa(question=question, context=context)
        return {"answer": res.get("answer", ""), "score": res.get("score", 0.0), "source_chunks": [h["text"] for h in hits]}
