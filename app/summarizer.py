# app/summarizer.py
from transformers import pipeline
from typing import List

class MultiSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        self.summarizer = pipeline("summarization", model=model_name, device=device)

    def summarize_chunks(self, chunks: List[str], max_length=150, min_length=40):
        summaries = []
        for c in chunks:
            try:
                out = self.summarizer(c, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(out[0]['summary_text'])
            except Exception as e:
                summaries.append(c[:max_length])
        return " ".join(summaries)

    def multi_level(self, text: str, chunker_fn, levels=("short","medium","long")):
        chunks = chunker_fn(text)
        short = self.summarize_chunks(chunks, max_length=60, min_length=20)
        med = self.summarize_chunks([short], max_length=180, min_length=60)
        long = self.summarize_chunks(chunks, max_length=300, min_length=100)
        return {"short": short, "medium": med, "long": long}
