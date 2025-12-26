from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    # CRITIQUE FIX: Switched to Sentence-Transformer model.
    # 'all-MiniLM-L6-v2' is better at capturing semantic nuance and intent 
    # than pure keyword matching, which is crucial for understanding emotional context.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings