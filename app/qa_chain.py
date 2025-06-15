from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()

def create_qa_pipeline(retriever):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")  
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
