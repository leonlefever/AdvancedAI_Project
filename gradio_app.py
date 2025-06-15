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

print("[INFO] Loading environment variables...")
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY is not set!")
else:
    print("[INFO] OPENAI_API_KEY loaded successfully.")

os.environ["OPENAI_API_KEY"] = api_key


# === 1. Load your data
try:
    print("[INFO] Loading dataset...")
    df = pd.read_csv("data/processed/listings_with_preds.csv")
    print(f"[INFO] Loaded {len(df)} listings.")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    raise

# === 2. Turn into documents
try:
    print("[INFO] Converting listings to documents...")
    docs = [
        Document(
            page_content=f"Listing {row.listing_id}: {row.title} in {row.subtitle}, price €{row.pred_price}",
            metadata={"listing_id": int(row.listing_id)}
        )
        for _, row in df.iterrows()
    ]
    print(f"[INFO] Created {len(docs)} documents.")
except Exception as e:
    print(f"[ERROR] Failed to create documents: {e}")
    raise

# === 3. Build embedding + retriever
try:
    print("[INFO] Initializing HuggingFace embeddings...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    print("[INFO] Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = vectorstore.as_retriever()
    print("[INFO] Vectorstore and retriever ready.")
except Exception as e:
    print(f"[ERROR] Failed to initialize embedding or vectorstore: {e}")
    raise

# === 4. Load LLM
try:
    print("[INFO] Loading OpenAI LLM (gpt-3.5-turbo)...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    print("[INFO] LLM and QA chain initialized.")
except Exception as e:
    print(f"[ERROR] Failed to initialize LLM or QA chain: {e}")
    raise

# === 5. Define the Gradio function
def answer_question(query):
    try:
        print(f"[INFO] Received query: {query}")
        response = qa.invoke(query)
        print("[INFO] Response generated.")
        answer = response["result"]
        sources = "\n\n".join([doc.page_content for doc in response["source_documents"]])
        return answer, sources
    except Exception as e:
        print(f"[ERROR] Error while answering question: {e}")
        return f"Error: {e}", ""



with gr.Blocks(title="Madrid Housing Assistant", css="footer {display: none !important}") as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown("## 🏠 Madrid Real Estate Assistant")
        gr.Markdown(
            "Ask anything about listings, neighborhoods, or property prices in Madrid. "
            "I use real data and AI to help you make smarter choices. 🤖🏡"
        )

        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="💬 Your Question",
                    placeholder="e.g. What's the average price in Salamanca?",
                    lines=2,
                    max_lines=4
                )

                example_questions = gr.Examples(
                    examples=[
                        ["Which neighborhoods are best for families?"],
                        ["Compare listing 1201 and 3400"],
                        ["What’s the estimated price of listing 8502?"],
                        ["What are the cheapest areas in Madrid?"]
                    ],
                    inputs=[query_input]
                )

                with gr.Row():
                    submit_btn = gr.Button("🔍 Ask")
                    clear_btn = gr.Button("🗑️ Clear")

                status_output = gr.Markdown("")  # Spinner/status text

            with gr.Column(scale=5):
                response_output = gr.Textbox(
                    label="✅ Assistant Response",
                    lines=10,
                    interactive=False
                )

        with gr.Accordion("📄 Retrieved Listings (Sources)", open=False):
            sources_output = gr.Textbox(
                lines=10,
                interactive=False,
                label="Top Matching Listings"
            )

        # Event bindings
        def wrapped_answer(query):
            print("[INFO] Wrapped function started")
            status = "⏳ Thinking..."
            answer, sources = answer_question(query)
            status = "✅ Answer ready."
            return answer, sources, status

        submit_btn.click(
            fn=wrapped_answer,
            inputs=[query_input],
            outputs=[response_output, sources_output, status_output]
        )


        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            inputs=[],
            outputs=[query_input, response_output, sources_output, status_output]
        )


# === 7. Launch
if __name__ == "__main__":
    print("[INFO] Launching app...")
    demo.launch()
