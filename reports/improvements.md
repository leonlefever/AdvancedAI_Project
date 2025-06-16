# Technical Report – Version 2: Improvements and Updates

## What I Improved Compared to the Previous Version

### 1. Modular Project Structure  
I restructured the project into clear and reusable modules, such as:
- `s01_embeddings.py` – for loading embedding models
- `s02_retriever.py` – for loading the FAISS index
- `s03_qa_chain.py` – for building the LangChain QA pipeline
- `s04_interface.py` – for the Gradio UI

This helped me keep responsibilities separated and made the code easier to debug, test, and explain.

---

### 2. Clear Separation of Embedding Usage  
I now separate:
- **Precomputed sentence embeddings** (stored in `.npy` files from `sentence-transformers`)
- From the **real-time HuggingFace embedding model** used by LangChain to encode user queries.

This allows me to efficiently match new queries to listings without recomputing everything.

---

### 3. Enhanced User Interface  
I redesigned the Gradio interface:
- Users can now either **type custom natural language questions** or
- **Use sliders and dropdowns** to describe a property and ask for its expected price.

I also applied a modern dark theme for better readability.

---

### 4. Token Cost and Query Control  
I added settings like `k=30` to control how many listings get retrieved per query.  
This gave me better control over OpenAI token usage and cost (which is important when using GPT-3.5).

---

### 5. Deployment-Ready Design  
I made the app easier to run:
- I added a `.env` file for API keys
- A detailed `README.md` with step-by-step instructions
- A clearly defined entry point in `gradio_app.py`

Now it works on any machine with just a few commands.

---

### 6. Improved Prompting and UX  
To guide users, I added predefined examples such as:
> *"What are the most expensive neighborhoods in Madrid?"*

This improved the user experience and made it easier to understand what the app can do.

---

### 7. Better Model Performance  
I tuned the XGBoost model and improved its predictive accuracy:
- Previous RMSE on log-price: ~0.222  
- New RMSE on log-price: **~0.216**

Even a small improvement here means more reliable price predictions.

---

### 8. Stronger Documentation  
The README now clearly explains:
- The step-by-step application flow
- Where embeddings are used
- How FAISS and LangChain work together
- How the Gradio app pulls everything together

This version is much more readable and beginner-friendly than my first report.

---

## Summary  
This version of the project is more readable and deployable.  

- Solid ML modeling (XGBoost + SHAP)
- Efficient vector search (FAISS + HuggingFace)
- User-friendly interaction (LangChain + Gradio)

It feels like a real product now—not just a demo.

