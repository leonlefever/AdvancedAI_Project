import os
import pandas as pd
import json
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

VECTORSTORE_DIR = "vectorstore"
LISTINGS_FILE = "data/processed/listings_with_preds.csv"

def load_vectorstore(embedding):
    print("[INFO] Loading FAISS vectorstore from disk...")
    vectorstore = FAISS.load_local(
        folder_path=VECTORSTORE_DIR,
        embeddings=embedding,
        index_name="madrid",
        allow_dangerous_deserialization=True
    )
    # if higher -> beter result but price goes up very fast for example: k = 100 means 0.03 euro per query. 
    # 100 queries a day means 3 euro a day!
    return vectorstore.as_retriever(search_kwargs={"k": 30})
