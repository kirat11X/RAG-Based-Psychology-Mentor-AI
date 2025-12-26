import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

# Page Config
st.set_page_config(page_title="Psychology Mentor", page_icon="🧠")

CHROMA_PATH = "chroma"

# UPDATED PROMPT: Adds guardrails against hallucinations
PROMPT_TEMPLATE = """
You are a warm, empathetic college mentor.

Your goal is to help the student based on the CHAT HISTORY and the provided CONTEXT.

CRITICAL INSTRUCTION:
The CONTEXT below is automatically retrieved from a database. It might be completely irrelevant to the current conversation. 
If the CONTEXT discusses topics (like pregnancy, severe disorders, specific case studies) that do NOT match the Student's current query or the CHAT HISTORY, you MUST IGNORE the CONTEXT.

Instead, respond naturally to the Student's latest message using the CHAT HISTORY.

CONTEXT:
{context}

CHAT HISTORY:
{history}

STUDENT'S LATEST MESSAGE: {question}

MENTOR'S RESPONSE:
"""

@st.cache_resource
def load_db():
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = OllamaLLM(model="mistral")
    return db, model

def query_rag(query_text, history, db, model):
    # Retrieve top 3 chunks
    results = db.similarity_search_with_score(query_text, k=3)
    
    # Check for relevance. If the best score is too low (distance too high), 
    # we might want to warn the model or just provide less context.
    # For now, we pass them but rely on the Prompt Guardrails to filter bad context.
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    history_text = "\n".join([f"Student: {q}\nMentor: {a}" for q, a in history])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, history=history_text, question=query_text)
    
    response_text = model.invoke(prompt)
    
    sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
    return response_text, list(set(sources))

# --- UI Layout ---
st.title("🧠 Psychology AI Mentor")
st.caption("Ask specific questions about psychology concepts found in your library.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load DB (Cached)
db, model = load_db()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare history list for RAG function
            # We pair strictly: (User, AI), (User, AI)...
            history_list = []
            msgs = st.session_state.messages
            for i in range(0, len(msgs) - 1):
                if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                    history_list.append((msgs[i]["content"], msgs[i+1]["content"]))
            
            response, sources = query_rag(prompt, history_list, db, model)
            
            st.markdown(response)
            
            # Show sources in an expander
            with st.expander("📚 Sources"):
                for source in sources:
                    st.write(f"- {source}")
    
    # Add assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": response})