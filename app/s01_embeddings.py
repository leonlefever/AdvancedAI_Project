from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import os

EMBEDDINGS_FILE = "vectorstore/embeddings.npy"

def load_model():

    print("[INFO] Loading SentenceTransformer model...")
    
    return SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

def load_embeddings():

    print("[INFO] Loading saved embeddings...")
    emb = np.load(EMBEDDINGS_FILE)
    print(f"[INFO] Loaded {emb.shape[0]} embeddings.")

    return emb

def load_langchain_embedding():

    print("[INFO] Loading HuggingFaceEmbeddings for LangChain...")
    
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
