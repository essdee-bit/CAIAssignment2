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

# Load pre-trained SentenceTransformer for embeddings and response generation
embedding_model = SentenceTransformer("all-mpnet-base-v2")  # Embedding model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Re-ranking model

def extract_text_from_pdfs(pdf_files):
    """Extract and clean text from multiple PDF files."""
    text_data = ""
    for file in pdf_files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                raw_text = page.extract_text()
                if raw_text:
                    cleaned_text = clean_text(raw_text)  # Clean extracted text
                    text_data += cleaned_text + " "  # Append cleaned text
    return text_data

def clean_text(text):
    """Clean extracted text by removing special characters, extra spaces, and non-informative sections."""
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^a-zA-Z0-9.,;:\n ]', '', text)  # Remove special characters
    text = re.sub(r'\b(UNITED STATES|SECURITIES AND EXCHANGE COMMISSION|FORM 10-K)\b', '', text, flags=re.IGNORECASE)  # Remove headers
    text = text.strip()  # Remove extra spaces
    return text

# Extract text from PDFs
document_text = extract_text_from_pdfs(pdf_files)

def chunk_text(text, chunk_size=512):
    """Chunk text into smaller segments for processing."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Split text into chunks
text_chunks = chunk_text(document_text)

# Initialize BM25 for keyword-based retrieval
bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])

# Generate embeddings for text chunks
embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks], dtype=np.float32)

# Initialize FAISS index for fast similarity search
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 index for similarity search
index.add(embeddings)  # Add embeddings to FAISS index

def retrieve_relevant_passage(query, top_k=5):
    """Retrieve the most relevant passage using FAISS and BM25."""
    query_embedding = embedding_model.encode(query).astype(np.float32)
    distances, idx = index.search(query_embedding.reshape(1, -1), top_k)

    # Convert L2 distance to similarity score (cosine similarity approximation)
    similarity_scores = [1 / (1 + d) * 100 for d in distances[0]]  # Convert to percentage
    faiss_results = [(text_chunks[i], similarity_scores[j]) for j, i in enumerate(idx[0])]

    # BM25 retrieval
    bm25_scores = bm25.get_scores(query.split())
    bm25_results = sorted(zip(text_chunks, bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]

    # Merge FAISS and BM25 results using a hybrid score
    hybrid_results = {}
    for passage, score in faiss_results:
        hybrid_results[passage] = score * 0.6  # FAISS weight
    for passage, score in bm25_results:
        if passage in hybrid_results:
            hybrid_results[passage] += score * 0.4  # BM25 weight
        else:
            hybrid_results[passage] = score * 0.4

    sorted_results = sorted(hybrid_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_results

def rerank_results(query, results, top_k=3):
    """Re-rank results using Cross-Encoder and return the top-k results."""
    pairs = [(query, passage) for passage, _ in results]
    scores = cross_encoder.predict(pairs)
    scores = F.softmax(torch.tensor(scores), dim=0).numpy() * 100
    ranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(res[0], res[1]) for res in ranked_results]

def get_rag_response(query):
    """Retrieve and re-rank results for a given query."""
    hybrid_results = retrieve_relevant_passage(query)
    ranked_results = rerank_results(query, hybrid_results)
    
    if not ranked_results or max(score for _, score in ranked_results) < 50:
        return "No highly relevant information found.", 0.0
    
    best_response, confidence = ranked_results[0]
    return best_response, confidence / 100  # Normalize confidence to 0-1 scale

def main():
    st.title("Financial Document Search")
    st.write("Search financial documents using AI-powered retrieval and ranking.")
    query = st.text_input("Enter your query:")
    
    if query:
        response, confidence = get_rag_response(query)
        st.subheader("Response")
        st.write(response)
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
