import argparse
import sys
import logging
import time
from collections import deque
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

# --- CONFIGURATION ---
CHROMA_PATH = "chroma"
# Relevance Threshold: 
# Chroma uses "L2 Distance" by default. 
# Lower distance = More similar. 
# 0.0 is an exact match. ~0.3-0.5 is usually good. > 1.0 is often irrelevant.
RELEVANCE_THRESHOLD = 0.7 
LOG_FILE = "chatbot_interaction.log"

# --- SAFETY & CRISIS CONFIGURATION ---
# Simple keyword matching for immediate safety interception.
# In production, use a dedicated classifier model (e.g., BERT-based sentiment analysis).
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "want to die", "end it all", "hurt myself", 
    "self-harm", "cutting", "overdose", "better off dead"
]

CRISIS_RESPONSE = """
⚠️ IMPORTANT: I hear that you are going through a very difficult time, but I am an AI, not a mental health professional. 

If you are in danger or need immediate help, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741
- Go to your nearest emergency room.

I cannot provide crisis support. Please reach out to a human who can help.
"""

# --- DISCLAIMER ---
LEGAL_DISCLAIMER = """
********************************************************************************
* DISCLAIMER:                                                                  *
* This AI is a supportive college mentor, NOT a mental health professional.    *
* It cannot diagnose, treat, or cure any mental health conditions.             *
* If you are in crisis, please seek professional help immediately.             *
********************************************************************************
"""

# --- UPDATED PROMPT WITH BOUNDARIES ---
PROMPT_TEMPLATE = """
You are a warm, empathetic college mentor AI. 
You are NOT a therapist, psychologist, or doctor. 

YOUR ROLE:
1. Provide supportive, non-medical advice based on the context provided.
2. Translate academic/textbook concepts into warm, human language.
3. MAINTAIN BOUNDARIES: If the user seems overly dependent or asks for a diagnosis, gently remind them you are an AI and suggest professional help.
4. IGNORE context if it is irrelevant to the user's feelings.

CONTEXT FROM BOOKS:
{context}

CHAT HISTORY:
{history}

---

USER'S QUESTION: {question}

MENTOR'S RESPONSE:
"""

def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    print(LEGAL_DISCLAIMER)
    print("--- Psychology Mentor CLI (Type 'quit' to stop) ---")
    print("--- Loading Brain (this may take a moment)... ---")
    
    # 1. Initialize components
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = OllamaLLM(model="mistral")
    
    # 2. Memory Setup
    # Increased history to 5 to allow for longer context retention as per critique
    history = deque(maxlen=5)
    
    print("Mentor: Hello! I'm here to listen and offer perspective from your library. How are you feeling?")

    while True:
        try:
            # 3. Get User Input
            query_text = input("\nStudent: ")
            
            # Handle exit commands
            if query_text.lower() in ['quit', 'exit', 'q']:
                print("Mentor: Take care of yourself. Remember to seek support if you need it. Bye.")
                break
            
            if not query_text.strip():
                continue

            # Log the user query
            logging.info(f"User Query: {query_text}")

            # --- SAFETY CHECK LAYER ---
            # Check for crisis keywords before hitting the LLM/Database
            if check_for_crisis(query_text):
                print(f"\nMentor: {CRISIS_RESPONSE}")
                logging.warning(f"Crisis Keyword Detected: {query_text}")
                continue
            # --------------------------

            # 4. Run the RAG pipeline
            response_text, sources, is_relevant = query_rag(query_text, history, db, model)
            
            # Log the response
            logging.info(f"Mentor Response: {response_text}")
            logging.info(f"Sources: {sources}")
            
            # 5. Update Memory
            history.append(f"Student: {query_text}\nMentor: {response_text}")

            # 6. Output Result
            print(f"\nMentor: {response_text}")
            
            # 7. Display Sources & Warnings
            if sources:
                print("\nSources:")
                for source in sources:
                    print(f" - {source}")
            
            if not is_relevant:
                print("\n(Note: This advice is general. I couldn't find specific passages in your library.)")
            
        except KeyboardInterrupt:
            print("\nMentor: Take care! Bye.")
            break

def check_for_crisis(text):
    """Simple keyword detection for crisis situations."""
    text_lower = text.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def query_rag(query_text: str, history: deque, db, model):
    # A. Search the DB with scores
    results = db.similarity_search_with_score(query_text, k=4)

    # B. Relevance Check
    is_relevant = True
    if not results or results[0][1] > RELEVANCE_THRESHOLD:
        is_relevant = False

    # C. Prepare Context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # D. Prepare History
    history_text = "\n".join(history)
    
    # E. Format Prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, history=history_text, question=query_text)

    # F. Generate Response
    response_text = model.invoke(prompt)
    
    # G. Extract unique sources
    sources = []
    if is_relevant:
        for doc, _score in results:
            source_id = doc.metadata.get("id", None)
            if source_id:
                sources.append(source_id)
    
    return response_text, list(set(sources)), is_relevant

if __name__ == "__main__":
    main()