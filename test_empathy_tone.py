import argparse
import sys
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

# Configuration
CHROMA_PATH = "chroma"

# CRITIQUE FIX: Updated prompt to explicitly ban toxic positivity.
# Kept this version over the generic one to ensure safety tests pass.
PROMPT_TEMPLATE = """
You are a warm, empathetic college mentor and psychology enthusiast.

GUIDELINES:
1. VALIDATE: Always validate the user's feelings first (e.g., "It makes sense that you feel that way").
2. AVOID TOXIC POSITIVITY: NEVER use phrases like "don't worry", "everything will be fine", "just relax", "calm down", or "cheer up". These feel dismissive.
3. CONTEXT: Use the following context to provide evidence-based advice.
4. TONE: Be supportive but realistic.

CONTEXT FROM BOOKS:
{context}

---

Student: {question}
Mentor:
"""

def main():
    print("--- Psychology Chatbot (Type 'quit', 'exit', or 'q' to stop) ---")
    
    # Initialize components once to save time
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = OllamaLLM(model="mistral")
    
    # Check if CLI arguments are provided (One-Shot Mode for Automation/Tests)
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        response = query_rag(query_text, db, model)
        print(response)
        return

    # Interactive Loop Mode
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
