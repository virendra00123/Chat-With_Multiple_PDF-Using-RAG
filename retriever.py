from sentence_transformers import SentenceTransformer
import numpy as np
from embedder import load_embeddings

# Use the same model as used for chunk embeddings!
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def embed_query(query, model_name=EMBEDDING_MODEL_NAME):
    """
    Converts a user query to a vector embedding.
    """
    model = SentenceTransformer(model_name)
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding

def search_similar_chunks(query, k=3):
    """
    Given a text query, finds the top-k most similar text chunks.
    Returns (list of chunk texts, list of metadata dicts).
    """
    # Load stored embeddings and metadata
    index, metadata = load_embeddings()
    
    # Embed the user query
    query_embedding = embed_query(query)
    
    # Search FAISS for top-k similar chunks
    D, I = index.search(np.array(query_embedding), k)
    # I is shape (1, k) – I[0] is list of indices
    
    # Retrieve matching chunk info
    top_chunks = []
    top_metadata = []
    for idx in I[0]:
        if idx < len(metadata):
            meta = metadata[idx]
            # For now, assuming you stored 'text' in metadata for simplicity:
            top_chunks.append(meta.get('text', '[TEXT NOT SAVED - see note in code]'))
            top_metadata.append(meta)
    
    return top_chunks, top_metadata