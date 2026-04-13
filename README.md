# 🚀 Multi-Engine FAQ System: Technical Documentation

A high-performance, modular NLP suite designed for high-accuracy FAQ retrieval across multiple domains (Medical, Legal, Finance, Tech, Education). The system utilizes a hybrid scoring mechanism combining Transformer-based semantics, lexical statistics, and linguistic structure (POS/NER).

---

## 🏗️ Project Architecture (Modular Design)

The system is split into specialized modules for maximum readability and scalability:

### 1. **The Core Orchestrator (`faq_system.py`)**
The main entry point. It manages:
*   **Initialization**: Bootstraps all 5 NLP engines and the domain classifier.
*   **Search Loop**: Processes user queries, runs parallel searches, and computes global rankings.
*   **Evaluation Mode**: Triggers a scientific benchmark of all engines using standard IR metrics.

### 2. **Embedding Engines (`engines/`)**
A swappable architecture where every engine inherits from `BaseEngine`:
*   **`tfidf_engine.py`**: Classic keyword-based matching (Lexical).
*   **`sbert_engine.py`**: Deep semantic understanding using `all-MiniLM-L6-v2` transformers.
*   **`gensim_engine.py`**: Static word embeddings using **GloVe** (Wiki-Gigaword) and **Word2Vec** (Google News).
*   **`infersent_engine.py`**: Supervised sentence embeddings using a Bi-LSTM architecture.

### 3. **Linguistic Utilities (`utils/`)**
*   **`preprocessor.py`**: A bridge to an external **C-based NLP Preprocessor**. It extracts Tokens, POS tags, and NER entities with high performance.
*   **`nlp_helpers.py`**: Contains the scoring logic for structural overlap and the **Answer Synthesis** engine.
*   **`dataset.py`**: Handles the custom parsing of the multi-domain FAQ text file.

### 4. **Intent Prediction (`classifier/`)**
*   **`domain.py`**: Uses a **Logistic Regression** model with TF-IDF vectorization to predict if a query belongs to a specific domain (e.g., "Legal"). This is used to "boost" relevant results and penalize domain-mismatched answers.

### 5. **Configuration (`config.py`)**
The "Master Control Room." You can adjust system weights (Alpha, Beta, Gamma) and domain keywords here without editing the main logic.

---

## 🧪 The Hybrid Scoring Algorithm

The system does not rely on a single score. For every FAQ candidate, a **Final Confidence Score** is calculated using this formula:

$$FinalScore = [(\alpha \cdot Sim) + (\beta \cdot POS) + (\gamma \cdot NER)] \cdot DomainPenalty$$

*   **$\alpha$ (Similarity - 0.5)**: The raw semantic distance from the embedding engines.
*   **$\beta$ (POS Match - 0.25)**: Measures how well the internal structure (nouns/verbs) of the query matches the target FAQ.
*   **$\gamma$ (NER Overlap - 0.25)**: Ensures that key entities (names, locations, specific terms) in the query are present in the answer.
*   **Domain Penalty (0.05x)**: If the system is 40%+ sure of a domain (e.g., Medical) and a result is from another domain (e.g., Tech), the score is cut by 95% to prevent domain-leakage.

---

## ✍️ Stage 6: Answer Synthesis (TL;DR)

The **Synthesized Summary** is a unique feature that prepares a human-readable "Long Answer" by aggregating the Top 3 results:

1.  **Domain Aggregation**: It groups the top 3 results by their predicted domain.
2.  **Robust Splitting**: It uses regex-based sentence splitting to extract the "core" of the answers while ignoring complex dots (like in URLs or abbreviations).
3.  **Synthesis Logic**:
    *   **Single Domain**: If all top matches are from the same domain, it combines the first two sentences of the #1 result to provide a deeper context.
    *   **Multi-Domain**: If results span multiple domains (e.g., query about "Legal and Health laws"), it takes the most relevant sentence from *each* domain and labels them (e.g., *"Regarding Medical: ... Regarding Legal: ..."*).

---

## 📊 Evaluation Metrics

When you run `/evaluate`, the system calculates two world-class metrics:

1.  **P@1 (Precision at 1)**: The percentage of test queries where the correct answer was ranked as the absolute #1 result.
2.  **MRR (Mean Reciprocal Rank)**: A measure of "how far down the list" the correct answer was. A score of 1.0 is perfect; 0.5 means the correct answer was usually at rank #2.

---

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execute the System**:
    ```bash
    python faq_system.py
    ```
3.  **Commands**:
    *   `[Any Question]`: Standard search + synthesis.
    *   `/evaluate`: Run the Comparative Accuracy Benchmarks.
    *   `/exit`: Stop the program.
