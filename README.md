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

### 2. Gradio Application

- **`gradio_app.py`**:
  - Implements a Gradio-based user interface for querying property data and retrieving predictions.
  - Uses FAISS for vector-based retrieval and HuggingFace embeddings for semantic search.

### 3. Data

- **Raw Data**: Located in `archive/`.
- **Processed Data**: Outputs from preprocessing and modeling are saved in `data/processed/`.

### 4. Vectorstore

- Stores embeddings and FAISS indices for efficient retrieval.

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
