from app.embeddings import load_langchain_embedding
from app.retriever import load_vectorstore
from app.qa_chain import create_qa_pipeline
from app.interface import create_ui

embedding = load_langchain_embedding()  
retriever = load_vectorstore(embedding)
qa_chain = create_qa_pipeline(retriever)

demo = create_ui(qa_chain)

if __name__ == "__main__":
    demo.launch()


