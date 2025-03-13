import streamlit as st
from cai_assignment import get_rag_response  # Import the function from backend script

def main():
    st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
    st.title("📊 Financial RAG Chatbot")
    
    # Financial Keywords for Validation
    financial_keywords = {"revenue", "profit", "net income", "earnings", "EPS", "cash flow", "expenses", "assets", "liabilities"}
    
    # User Input
    user_query = st.text_input("Ask a financial question:", "")
    
    if st.button("Get Answer") and user_query:
        # Retrieve response and confidence score from the backend
        response, confidence = get_rag_response(user_query)
        
        # Output-side guardrail: Handle low-confidence responses
        if confidence < 0.5:
            response = "⚠️ The model is unsure about this response. Please verify with official financial reports."
        else:
            # Check if response contains key financial terms
            if not any(keyword in response.lower() for keyword in financial_keywords):
                response = "⚠️ This response may not contain relevant financial details. Please verify with official sources."
            
            # Check for uncertainty phrases
            uncertainty_phrases = ["I think", "maybe", "it is likely", "possibly"]
            if any(phrase in response.lower() for phrase in uncertainty_phrases):
                response += "\n\n⚠️ This answer contains uncertainty. Cross-check with financial statements."
        
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

if __name__ == "__main__":
    main()


