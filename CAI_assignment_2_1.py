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
 
# File paths to financial documents
pdf_files = ["aapl-20220924.pdf", "aapl-20230930.pdf"]
 
# Load pre-trained models
embedding_model = SentenceTransformer("all-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
def extract_text_from_pdfs(pdf_files):
    """Extract and clean text from multiple PDF files."""
    text_data = ""
    for file in pdf_files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found. Skipping.")
            continue
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                raw_text = page.extract_text()
                if raw_text:
                    cleaned_text = clean_text(raw_text)
                    text_data += cleaned_text + " "
    return text_data
 
def clean_text(text):
    """Clean extracted text by removing special characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:\n ]', '', text)
    text = text.strip()
    return text
 
# Extract text from PDFs
document_text = extract_text_from_pdfs(pdf_files)
 
def chunk_text(text, chunk_size=512):
    """Chunk text into smaller segments."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
 
# Split text into chunks
text_chunks = chunk_text(document_text)
 
# Initialize BM25 for keyword-based retrieval
bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])
 
# Generate embeddings for text chunks
embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks], dtype=np.float32)
 
# Initialize FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
 
def retrieve_relevant_passage(query, top_k=5):
    """Retrieve the most relevant passage using FAISS and BM25."""
    query_embedding = embedding_model.encode(query).astype(np.float32)
    distances, idx = index.search(query_embedding.reshape(1, -1), top_k)
    similarity_scores = [1 / (1 + d) * 100 for d in distances[0]]
    faiss_results = [(text_chunks[i], similarity_scores[j]) for j, i in enumerate(idx[0])]
    bm25_scores = bm25.get_scores(query.split())
    bm25_results = sorted(zip(text_chunks, bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
    hybrid_results = {}
    for passage, score in faiss_results:
        hybrid_results[passage] = score * 0.6
    for passage, score in bm25_results:
        hybrid_results[passage] = hybrid_results.get(passage, 0) + score * 0.4
    return sorted(hybrid_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
 
def rerank_results(query, results, top_k=3):
    """Re-rank results using Cross-Encoder."""
    pairs = [(query, passage) for passage, _ in results]
    scores = cross_encoder.predict(pairs)
    scores = F.softmax(torch.tensor(scores), dim=0).numpy() * 100
    return sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:top_k]
 
def filter_hallucinations(response, confidence):
    """Filter responses with low confidence."""
    if confidence < 40:
        return "The system is unsure about the answer. Please check the financial reports directly."
    return response
 
def get_rag_response(query):
    """Retrieve and re-rank results."""
    hybrid_results = retrieve_relevant_passage(query)
    ranked_results = rerank_results(query, hybrid_results)
    if not ranked_results or max(score for _, score in ranked_results) < 50:
        return "No highly relevant information found.", 0.0
    best_response, confidence = ranked_results[0]
    return filter_hallucinations(best_response[0], confidence), min(confidence / 100, 1.0)
 
def is_valid_query(query):
    """Validate user input to prevent irrelevant queries."""
    banned_words = ["scam", "hack", "illegal"]
    if len(query) < 5:
        return False, "Query is too short. Please provide more details."
    if any(word in query.lower() for word in banned_words):
        return False, "This query contains restricted words."
    return True, ""
 
def main():
    st.title("Financial RAG Chatbot")
    user_query = st.text_input("Enter your financial question:")
    if st.button("Get Answer"):
        if user_query:
            valid, message = is_valid_query(user_query)
            if not valid:
                st.error(message)
            else:
                response, confidence = get_rag_response(user_query)
                st.write(f"**Answer:** {response}")
                st.write(f"**Confidence Score:** {confidence:.2f}")
        else:
            st.warning("Please enter a query.")
 
if __name__ == "__main__":
    main()
