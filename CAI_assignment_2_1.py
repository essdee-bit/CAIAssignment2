import os
import pdfplumber
import faiss
import numpy as np
import re
import torch
import torch.nn.functional as F
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import threading

# File paths to financial documents
pdf_files = ["aapl-20220924.pdf", "aapl-20230930.pdf"]

@st.singleton
def load_resources():
    """Load models and prepare embeddings and FAISS index."""
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    document_text = extract_text_from_pdfs(pdf_files)
    text_chunks = chunk_text(document_text)
    bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])
    embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks], dtype=np.float32)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    return embedding_model, cross_encoder, text_chunks, bm25, index

def extract_text_from_pdfs(pdf_files):
    """Extract and clean text from multiple PDF files."""
    text_data = ""
    for file in pdf_files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                raw_text = page.extract_text()
                if raw_text:
                    text_data += clean_text(raw_text) + " "
    return text_data

def clean_text(text):
    """Clean extracted text by removing special characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:\n ]', '', text)
    text = re.sub(r'\b(UNITED STATES|SECURITIES AND EXCHANGE COMMISSION|FORM 10-K)\b', '', text, flags=re.IGNORECASE)
    return text.strip()

def chunk_text(text, chunk_size=512):
    """Chunk text into smaller segments for processing."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def retrieve_relevant_passage(query, embedding_model, text_chunks, bm25, index, top_k=5):
    """Retrieve the most relevant passage using FAISS and BM25."""
    query_embedding = embedding_model.encode(query).astype(np.float32)
    distances, idx = index.search(query_embedding.reshape(1, -1), top_k)
    similarity_scores = [1 / (1 + d) * 100 for d in distances[0]]
    faiss_results = [(text_chunks[i], similarity_scores[j]) for j, i in enumerate(idx[0])]
    
    bm25_scores = bm25.get_scores(query.split())
    bm25_results = sorted(zip(text_chunks, bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    hybrid_results = {passage: score * 0.6 for passage, score in faiss_results}
    for passage, score in bm25_results:
        hybrid_results[passage] = hybrid_results.get(passage, 0) + score * 0.4
    
    return sorted(hybrid_results.items(), key=lambda x: x[1], reverse=True)[:top_k]

def rerank_results(query, results, cross_encoder, top_k=3):
    """Re-rank results using Cross-Encoder."""
    pairs = [(query, passage) for passage, _ in results]
    scores = cross_encoder.predict(pairs)
    scores = F.softmax(torch.tensor(scores), dim=0).numpy() * 100
    return sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:top_k]

def get_rag_response(query, embedding_model, cross_encoder, text_chunks, bm25, index):
    """Retrieve and re-rank results for a given query."""
    hybrid_results = retrieve_relevant_passage(query, embedding_model, text_chunks, bm25, index)
    ranked_results = rerank_results(query, hybrid_results, cross_encoder)
    
    if not ranked_results or max(score for _, score in ranked_results) < 50:
        return "No highly relevant information found.", 0.0
    
    best_response, confidence = ranked_results[0]
    return best_response, confidence / 100

def load_resources_threaded():
    """Run the load_resources function in a separate thread."""
    thread = threading.Thread(target=load_resources)
    thread.start()
    return thread

def main():
    st.title("Financial Document Search")
    st.write("Search financial documents using AI-powered retrieval and ranking.")
    
    with st.spinner("Loading resources..."):
        # Load resources in a separate thread
        thread = load_resources_threaded()
        thread.join()  # Wait for the thread to finish
        
        # Resources are loaded after the thread completes
        embedding_model, cross_encoder, text_chunks, bm25, index = load_resources()
    
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Processing query..."):
            response, confidence = get_rag_response(query, embedding_model, cross_encoder, text_chunks, bm25, index)
        st.subheader("Response")
        st.write(response)
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
