# Technical Reflection & Possible Improvements

## Key Limitations

### 1. Embedding Numerical Values as Text

Right now, structured numeric values like `sq_mt_built = 80` or `pred_price = €200,000` are embedded **as plain text** inside sentences (e.g., "Predicted price: €200,000").

- Problem: Sentence transformers are trained for **semantic language**, not precise numeric comparison.
- Result: The model may embed “€100,000” and “€500,000” similarly if the surrounding context is close.
- This makes the embeddings less reliable for tasks like price filtering, size ranges, or statistical reasoning.

**Possible Improvements:**

- Use **hybrid search**: combine semantic vector similarity with structured filters (e.g., via metadata filtering or a ranking function).
- Add **numeric features separately** to the FAISS vector using concatenated embeddings or multimodal encoders.
- Use newer models that support **numerical-aware embeddings** (e.g., GTE, TabFormer, etc.).

---

### 2. FAISS Limitation: Top-k Fixed Search

Currently, my retriever runs FAISS with a fixed `k=30`, meaning only 30 listings are retrieved per query.

- Problem: The assistant cannot compute **averages**, **counts**, or **global statistics** (e.g., “What’s the average price in Salamanca?”) if the relevant listings are **not in the top 30 vectors**.
- Because FAISS only returns top-k based on cosine similarity to the **embedded question**, many relevant rows may be ignored.

**Possible Improvements:**

- Add a second non-FAISS retrieval layer (e.g., Pandas filtering) when the question contains statistical intent.
- Auto-detect **aggregation queries** using keyword detection or LLM prompt parsing.
- Allow **dynamic `k` adjustment** for specific question types, balancing cost vs. coverage.
- Precompute and cache **neighborhood-level statistics** to serve instantly.

---

### 3. Embedding Update Workflow Is Static

Currently, embeddings are computed once and saved (`embeddings.npy`, `madrid.faiss`, etc.).

- New listings require **manual re-embedding** and re-saving.
- FAISS index must be rebuilt every time, which is inefficient.

**Possible Improvements:**

- Implement **incremental embedding**: only embed and insert new listings.
- Use **FAISS add method** (`index.add(...)`) to insert new vectors without full rebuild.
- Maintain an updated listing hash or ID tracker to detect what’s changed.

---

### 4. No Hybrid Filtering (Metadata Not Used in Ranking)

Even though each LangChain `Document` has metadata (e.g., `listing_id`, `neighborhood_id`), this is **not used to filter or prioritize results**.

- This limits precision — e.g., asking for “homes in Salamanca with a terrace” doesn’t guarantee only those will appear.
- All ranking is done purely based on vector proximity to the embedded question.

**Possible Improvements:**

- Use LangChain’s **metadata filters** (`retriever.search_kwargs={"filter": ...}`).
- Add a **post-ranking step** that filters or reranks results using structured logic (e.g., Pandas conditions).

---

### 5. Cost Risk with LLM + High `k`

While using `k=30` gives better results, it increases the prompt size sent to GPT-3.5.

- This means higher **OpenAI costs** if used at scale.
- Example: 30 listings × 500 tokens per listing = 15,000 tokens/query → ~$0.05 per question.

**Possible Improvements:**

- Automatically **truncate or summarize** long listings before sending to LLM.
- Dynamically reduce `k` based on query type or estimated answer complexity.
- Add a **"cost-saving mode"** that uses local filters or a cheaper LLM (e.g., Mistral or Llama 3).

---

## Summary Table

| Problem Area               | Cause                           | Suggested Fix                            |
| -------------------------- | ------------------------------- | ---------------------------------------- |
| Numeric values as text     | Language-only embeddings        | Use hybrid or numeric-aware embeddings   |
| Top-k FAISS limits         | FAISS returns only k vectors    | Add Pandas/statistical fallback layer    |
| Static embedding workflow  | No dynamic update               | Use incremental indexing                 |
| No metadata filtering      | Filters not applied to results  | Use LangChain metadata filters           |
| Token costs scale with `k` | Large retrieval set sent to LLM | Summarize docs or adjust `k` dynamically |

---

## Final Thoughts

The current embedding setup works well for semantic question-answering, but it lacks flexibility for numeric filtering, aggregations, and dynamic updates. If I were to productionize this system, I would shift toward **hybrid semantic + symbolic reasoning**, and invest in **smarter retrieval logic** that doesn’t rely purely on FAISS similarity.
