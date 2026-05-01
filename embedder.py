from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 
FAISS_PATH = "vector_store/faiss.index"
META_PATH = "vector_store/metadata.json"

def embed_chunks(chunks, model_name=EMBEDDING_MODEL_NAME):
    """
    Converts a list of text chunks into vector embeddings using Hugging Face model.
    """
    model = SentenceTransformer(model_name)  # Now loads from Hugging Face automatically
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings


def store_embeddings(embeddings, metadata, faiss_path=FAISS_PATH, meta_path=META_PATH):
    """
    Stores embeddings in a FAISS index and saves metadata mapping.
    """
    if not os.path.exists(os.path.dirname(faiss_path)):
        os.makedirs(os.path.dirname(faiss_path))
        
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, faiss_path)
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)

def load_embeddings(faiss_path=FAISS_PATH, meta_path=META_PATH):
    """
    Loads FAISS index and metadata file.
    Returns (faiss_index, metadata_list)
    """
    index = faiss.read_index(faiss_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata
