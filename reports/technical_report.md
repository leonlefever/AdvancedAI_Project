# Technical Report: AI-Powered Real Estate Price Assistant

## 1. Introduction

**Problem Statement:**  
Real estate pricing is highly complex and influenced by many factors such as location, size, age, and features. For home buyers, sellers, and agents, interpreting these factors to understand fair pricing is difficult without expertise.

**Motivation:**  
I wanted to create an intelligent assistant that can:

- Predict property prices based on structured data.
- Explain what drives a listing's value.
- Answer natural language questions using real housing data.

**Chosen Approach:**  
I combined supervised machine learning (XGBoost regression), explainability (SHAP), and retrieval-augmented generation (RAG) using a local embedding model and GPT-3.5 via OpenAI to create a chat assistant.

**Repo:**  
[GitHub Repository](https://github.com/leonlefever/AdvancedAI_Project.git)

---

## 2. Data

**Source:**  
The dataset is a cleaned version of a Madrid housing dataset (~22,000 rows, 60+ features).

**Preprocessing Steps:**

- Dropped columns with >80% missing values.
- Imputed remaining missing values (mode, median, boolean False).
- Encoded categorical features (e.g., house type).
- Created engineered features like `log_price`, `building_age`, etc.

**Challenges:**

- Many features like `street_number`, `door`, `portal` were 100% missing.
- Missing latitude/longitude limited use of map-based models.
- Address inconsistencies required me to ignore geolocation.

---

## 3. Model & Methods

**Price Prediction:**  
I trained an XGBoost Regressor on selected numeric and categorical features. I evaluated it using RMSE on log-transformed price values and achieved an RMSE of ~0.222 (log-scale).

**Explainability (SHAP):**  
For each predicted price, I used SHAP to extract the top 3 most impactful features. These were stored in a JSON field (`shap_top3`) for embedding purposes.

**Semantic Embeddings:**  
Using the MiniLM model (384-dim), I embedded a textual summary per listing (title, area, top-3 SHAP features, predicted price).

**Vector Store + LLM:**

- I used FAISS to build an index of listing vectors.
- RetrievalQA (LangChain) was set up using `gpt-3.5-turbo`.

---

## 4. Results & Evaluation

**Model Performance:**

- XGBoost RMSE (log-scale): 0.222
- Qualitatively, predictions are plausible and consistent across neighborhoods.

**Insights:**

- Size (`sq_mt_built`), terrace presence, and lift availability are top drivers.
- Chamartín, Salamanca, and Chamberí consistently rank as more expensive.

**LLM QA Results:**

- Returns contextual answers like "Homes in Chamartín are more expensive due to amenities and location."

---

## 5. Contributions

**Leon:**

- Data cleaning, feature engineering, XGBoost training, SHAP implementation.
- Created all embedding + FAISS logic.
- Set up RetrievalQA + Gradio interface.
- Integrated OpenAI API.

**GenAI Use:**

- ChatGPT helped debug error messages, optimize SHAP logic with embeddings, helped with LangChain.

**Online Resources:**

- LangChain documentation.
- XGBoost + SHAP official docs.
- MiniLM from Hugging Face.

---

## 6. Challenges & Future Work

**Challenges:**

- FAISS installation and GPU support was inconsistent.
- Vectorstore embeddings.
- SHAP value explanations slowed down large-scale embedding.
- Gradio deployment had issues due to LangChain updates.

**Future Plans (if possible):**

- Extend to multi-city dataset.
- Let users upload listings and get real-time pricing + drivers.
