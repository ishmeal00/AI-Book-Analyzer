# streamlit_app.py
import streamlit as st
from app.io import read_file
from app.processor import clean_text, detect_language, chunk_text
from app.embeddings import Embedder
from app.indexer import FaissIndexer
from app.summarizer import MultiSummarizer
from app.qa import RetrievalQA
from app.utils import extract_keywords, extract_entities
from app.visualize import build_knowledge_graph, generate_wordcloud
import os, json

st.set_page_config(page_title="AI Book Analyzer", layout="wide")
st.title("📚 AI Book Analyzer — Advanced (Local Demo)")

uploaded = st.file_uploader("Upload a file (PDF/TXT/DOCX)", type=["pdf","txt","docx"])
model_device = -1  # change if you have GPU: e.g., device=0

if uploaded:
    os.makedirs("output", exist_ok=True)
    temp_path = os.path.join("data", uploaded.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Reading file...")
    text = read_file(temp_path)
    text = clean_text(text)
    lang = detect_language(text)
    st.write(f"Detected language: **{lang}**")
    st.text_area("Text preview (first 2000 chars)", text[:2000], height=200)

    st.info("Embedding and indexing...")
    chunks = chunk_text(text, chunk_size_words=300, overlap=60)
    embedder = Embedder()
    vecs = embedder.embed(chunks)
    indexer = FaissIndexer(dim=vecs.shape[1])
    indexer.add(vecs, chunks, metas=[{"chunk_id": i} for i in range(len(chunks))])
    st.success(f"Indexed {len(chunks)} chunks.")

    st.info("Summarizing (multi-level)...")
    summarizer = MultiSummarizer()
    summaries = summarizer.multi_level(text, chunker_fn=lambda t: chunk_text(t, 300, 60))
    st.subheader("🔹 Short summary")
    st.write(summaries["short"])
    st.subheader("🔹 Medium summary")
    st.write(summaries["medium"])
    st.subheader("🔹 Detailed summary")
    st.write(summaries["long"])
    # save summaries
    with open("output/summary_short.txt", "w", encoding="utf-8") as f:
        f.write(summaries["short"])
    with open("output/summary_medium.txt", "w", encoding="utf-8") as f:
        f.write(summaries["medium"])
    with open("output/summary_long.txt", "w", encoding="utf-8") as f:
        f.write(summaries["long"])

    st.info("Extracting keywords & entities...")
    keywords = extract_keywords(text, max_keywords=25)
    entities = extract_entities(text)
    st.write("Top keywords:", keywords[:15])
    st.write("Named entities sample:", entities[:10])
    with open("output/keywords.json", "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)

    st.info("Generating visualizations...")
    generate_wordcloud(text, save_path="output/wordcloud.png")
    build_knowledge_graph(keywords[:20], entities[:20], save_path="output/knowledge_graph.png")
    st.image("output/wordcloud.png", caption="WordCloud", use_column_width=True)
    st.image("output/knowledge_graph.png", caption="Knowledge Graph", use_column_width=True)

    qa = RetrievalQA(embedder, indexer)
    st.subheader("Ask questions about the book (Retrieval QA)")
    q = st.text_input("Enter your question")
    if q:
        with st.spinner("Searching and answering..."):
            ans = qa.answer(q, top_k=6)
            st.write("**Answer:**", ans["answer"])
            st.write("**Score:**", ans.get("score"))
            st.write("**Source snippets:**")
            for i, s in enumerate(ans["source_chunks"]):
                st.markdown(f"**Snippet {i+1}:** {s[:600]}...")
    st.success("All done — results saved in `output/` folder.")
