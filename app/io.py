# app/io.py
import PyPDF2
import docx
import os
import pdfplumber

def read_pdf(path: str) -> str:
    text = ""
    # try pdfplumber first for better extraction
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    # fallback to PyPDF2
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print("PDF reading error:", e)
    return text

def read_txt(path: str, encoding="utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()

def read_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".txt":
        return read_txt(path)
    if ext in [".docx", ".doc"]:
        return read_docx(path)
    raise ValueError("Unsupported file type: " + ext)
