import gradio as gr
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = api_key
# === 1. Load your data
df = pd.read_csv("data/processed/listings_with_preds.csv")


# === 2. Turn into documents
docs = [
    Document(
        page_content=f"Listing {row.listing_id}: {row.title} in {row.subtitle}, price €{row.pred_price}",
        metadata={"listing_id": int(row.listing_id)}
    )
    for _, row in df.iterrows()
]

# === 3. Build embedding + retriever
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# === 4. Load LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === 5. Define the Gradio function
def answer_question(query):
    try:
        response = qa.invoke(query)
        return response["result"]
    except Exception as e:
        return f"Error: {e}"

# === 6. Gradio UI
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question about Madrid housing"),
    outputs=gr.Textbox(label="Assistant Response"),
    title="Real Estate Assistant",
    description="Ask about pricing, features, comparisons, and more."
)

# === 7. Launch
if __name__ == "__main__":
    demo.launch()
