from app.s01_embeddings import load_langchain_embedding
from app.s02_retriever import load_vectorstore
from app.s03_qa_chain import create_qa_pipeline
from app.s04_interface import create_ui

embedding = load_langchain_embedding()  
retriever = load_vectorstore(embedding)
qa_chain = create_qa_pipeline(retriever)

demo = create_ui(qa_chain)

if __name__ == "__main__":
    demo.launch()


