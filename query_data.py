import argparse
import sys
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

# Configuration
CHROMA_PATH = "chroma"

# Updated prompt to handle conversation flow better
PROMPT_TEMPLATE = """
You are a warm, empathetic college mentor and psychology enthusiast.
Your goal is to provide supportive, evidence-based advice based on the psychological texts provided.

CONTEXT FROM BOOKS:
{context}

---

USER'S QUESTION: {question}

MENTOR'S RESPONSE:
"""

def main():
    print("--- Psychology Chatbot (Type 'quit', 'exit', or 'q' to stop) ---")
    
    # Initialize components once to save time
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = OllamaLLM(model="mistral")
    
    while True:
        try:
            query_text = input("\nStudent: ")
            if query_text.lower() in ['quit', 'exit', 'q']:
                print("Mentor: Take care of yourself! Bye.")
                break
            
            if not query_text.strip():
                continue

            response = query_rag(query_text, db, model)
            print(f"\nMentor: {response}")
            
        except KeyboardInterrupt:
            print("\nMentor: Take care! Bye.")
            break

def query_rag(query_text: str, db, model):
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    response_text = model.invoke(prompt)

    return response_text

if __name__ == "__main__":
    main()