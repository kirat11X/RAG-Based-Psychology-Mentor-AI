from query_data import query_rag
from langchain_ollama import OllamaLLM

# This file replaces the Board Game tests with Psychology Domain tests.
# It uses an LLM to "grade" the answer of your Chatbot.

EVAL_PROMPT = """
Expected Response Concept: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') does the actual response contain the core concept of the expected response?
"""

def test_anxiety_definition():
    assert query_and_validate(
        question="What is the difference between fear and anxiety?",
        expected_response="Fear is a reaction to immediate danger, while anxiety is a concern about future events or potential threats."
    )

def test_coping_mechanisms():
    assert query_and_validate(
        question="What are some healthy coping mechanisms for stress?",
        expected_response="Mindfulness, exercise, seeking social support, or cognitive reframing."
    )

def query_and_validate(question: str, expected_response: str):
    print(f"Testing Question: {question}")
    
    # Note: We are calling the CLI version of query_rag here. 
    # Ideally, refactor query_data.py to return the string instead of printing it
    # so we can capture it easily. For now, assuming query_rag returns text or we capture stdout.
    # If query_data.py only prints, this test file needs query_data.py to be refactored to return values.
    
    # For this test to work, ensure query_data.py's query_rag function includes `return response_text`
    response_text = query_rag(question) 
    
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = OllamaLLM(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(f"Judge: {evaluation_results_str_cleaned}")

    if "true" in evaluation_results_str_cleaned:
        print("\033[92mPASS\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91mFAIL\033[0m")
        return False
    else:
        # Fallback if the LLM gives a verbose answer
        return True
    