import streamlit as st

def main():
    st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
    st.title("📊 Financial RAG Chatbot")
    
    # User Input
    user_query = st.text_input("Ask a financial question:", "")
    
    if st.button("Get Answer") and user_query:
        # Placeholder for retrieval and response generation
        response, confidence = get_rag_response(user_query)
        
        # Display Response
        st.subheader("Answer:")
        st.write(response)
        st.text(f"Confidence Score: {confidence:.2f}")
        
        # Optional: Query history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append((user_query, response, confidence))
        
    # Display Query History
    if "history" in st.session_state:
        st.subheader("Query History")
        for query, resp, conf in st.session_state.history[-5:]:
            st.write(f"**Q:** {query}")
            st.write(f"**A:** {resp}")
            st.text(f"Confidence: {conf:.2f}")
            st.markdown("---")

def get_rag_response(query):
    # Placeholder function for RAG retrieval and response generation
    # Replace this with actual implementation
    return "Sample financial answer", 0.85

if __name__ == "__main__":
    main()

