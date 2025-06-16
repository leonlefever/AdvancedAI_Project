# Advanced AI Project

## Overview

This project is part of the Advanced AI course and focuses on analyzing, preprocessing, and modeling real estate data to predict property prices and provide insights using machine learning and AI techniques. The project includes data exploration, preprocessing, modeling, and deployment of a Gradio-based application for user interaction.

## Project Structure

The project is organized into the following main components:

### 1. Notebooks

- **`01_data_exploration.ipynb`**:

  - Explores the raw dataset, identifies missing values, and provides an overview of the data types and distributions.
  - Outputs include null value statistics and unique value counts for each column.

- **`02_preprocessing.ipynb`**:

  - Handles missing values, feature engineering, and data transformation.
  - Includes steps like imputing missing values, creating new features (e.g., `price_per_m2`), and log-transforming skewed columns.

- **`03_modelling.ipynb`**:
  - Builds machine learning models to predict property prices.
  - Includes SHAP (SHapley Additive exPlanations) analysis for feature importance and embedding generation for retrieval tasks.

### 2. Gradio-Based Application

This application allows users to interact with real estate data using natural language queries.

- **`gradio_app.py`**  
  - Entry point that launches the Gradio interface.
  - Orchestrates the embedding model, retriever, LLM pipeline, and UI.

---

### 3. Data Structure

- **Raw Data**:  
  Stored in the `archive/` directory.

- **Processed Data**:  
  Cleaned, transformed, and enriched datasets are saved in `data/processed/`.

- **Vectorstore**:  
  Contains:
  - FAISS index (`madrid.faiss`)
  - Serialized metadata (`madrid.pkl`)
  - Embedding matrix (`embeddings.npy`)
  - Listing IDs (`ids.npy`)  
  Located in the `vectorstore/` directory for fast vector search.
  
  Note: The Embedding matrix (`embeddings.npy`) and Listing IDs (`ids.npy`) are currently not in use. 
    they are loaded in the backend (s01_embeddings.py) for manual testing later on.

---

### 4. `app/` Module Breakdown

This folder contains the modular backend logic for the Gradio app:

| File                | Responsibility                                                            |
|---------------------|---------------------------------------------------------------------------|
| `s01_embeddings.py` | Loads the HuggingFace embedding model used for semantic encoding.         |
| `s02_retriever.py`  | Loads the FAISS index and exposes it as a retriever with metadata support.|
| `s03_qa_chain.py`   | Creates the LangChain `RetrievalQA` chain using the OpenAI LLM.           |
| `s04_interface.py`  | Constructs the Gradio UI and formats both input and output presentation.  |

---

### 🔄 Application Flow

1. **Embeddings**:  
   `s01_embeddings.py` loads the sentence-transformers model for vector encoding.

2. **Vectorstore**:  
   `s02_retriever.py` loads the FAISS index and maps it to listing metadata.

3. **QA Pipeline**:  
   `s03_qa_chain.py` connects the retriever to OpenAI's LLM via LangChain.

4. **User Interface**:  
   `s04_interface.py` sets up the Gradio Blocks layout, input form, and result rendering.

---



### 🔄 App Flow

1. `s01_embeddings.py` loads the embedding model.
2. `s02_retriever.py` loads the FAISS vector index with metadata.
3. `s03_qa_chain.py` connects the retriever to the OpenAI LLM.
4. `s04_interface.py` wraps everything into a chatbot UI via Gradio.

---

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   - Create a `.env` file and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key
     ```
   - _Note_: This project uses OpenAI’s paid API. If testing, you may use the shared key from Toledo (uploaded separately).  
     ⚠️ **Each prompt costs ~$0.0001**, so avoid infinite loops.

4. Run Notebooks:

   - Start with `01_data_exploration.ipynb` to explore the data.
   - Proceed to `02_preprocessing.ipynb` for data cleaning and feature engineering.
   - Use `03_modelling.ipynb` to train models and generate embeddings.

5. Run Gradio App:
   - Start the Gradio application:
     ```bash
     python gradio_app.py
     ```
   - Access the app in your browser to query property data and view predictions.
