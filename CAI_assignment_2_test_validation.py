import json

def test_rag_chatbot():
    test_cases = [
        {"query": "What was the net income of Company X in 2023?", "expected": "high-confidence"},
        {"query": "What is the expected revenue growth for Company X next year?", "expected": "low-confidence"},
        {"query": "What is the capital of France?", "expected": "irrelevant"}
    ]
    
    results = []
    
    for test in test_cases:
        query = test["query"]
        expected = test["expected"]
        
        response, confidence = get_rag_response(query)
        
        # Determine test outcome
        if confidence > 0.7:
            outcome = "high-confidence"
        elif confidence < 0.5:
            outcome = "low-confidence"
        else:
            outcome = "medium-confidence"
        
        if "unsure" in response.lower() or "verify" in response.lower():
            guardrail_triggered = True
        else:
            guardrail_triggered = False
        
        results.append({
            "query": query,
            "expected": expected,
            "actual": outcome,
            "confidence": confidence,
            "response": response,
            "guardrail_triggered": guardrail_triggered
        })
    
    # Save results to a file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Test results saved to test_results.json")

def get_rag_response(query):
    # Placeholder function (Replace with actual chatbot function)
    test_responses = {
        "What was the net income of Company X in 2023?": ("Net income was $5M.", 0.85),
        "What is the expected revenue growth for Company X next year?": ("⚠️ The model is unsure. Please verify with official financial reports.", 0.4),
        "What is the capital of France?": ("⚠️ This question is not relevant to financial data.", 0.2)
    }
    return test_responses.get(query, ("No response available.", 0.0))

if __name__ == "__main__":
    test_rag_chatbot()

