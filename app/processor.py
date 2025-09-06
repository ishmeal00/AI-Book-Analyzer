# app/processor.py
from langdetect import detect
import re
from typing import List

def detect_language(text: str) -> str:
    try:
        return detect(text[:2000])
    except Exception:
        return "en"

def clean_text(text: str) -> str:
    # basic cleaning
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def chunk_text(text: str, chunk_size_words: int = 250, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i+chunk_size_words]
        chunks.append(" ".join(chunk))
        i += chunk_size_words - overlap
    return chunks
