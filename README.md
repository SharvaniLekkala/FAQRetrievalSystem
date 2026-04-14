# 🚀 Multi-Engine FAQ System: Technical Documentation

A high-performance, modular NLP suite designed for high-accuracy FAQ retrieval across multiple domains (Medical, Legal, Finance, Tech, Education). The system utilizes a hybrid scoring mechanism combining Transformer-based semantics, lexical statistics, and linguistic structure (POS/NER).

---

## 🏁 Performance Breakthrough: 69% → 92%

The system has undergone a major optimization journey to reach professional-grade accuracy on paraphrased user queries.

### 📊 Comparative Metrics
| Model Configuration | P@1 (Top-1 Accuracy) | MRR (Ranking Quality) |
| :--- | :--- | :--- |
| **Initial Baseline (MiniLM)** | 69.84% | 0.7981 |
| **MPNet Base Model** | 76.19% | 0.8384 |
| **MPNet (1st Round Tuning)** | 87.30% | 0.9233 |
| **⭐ Final Optimized (L12 Deep Tuning)** | **92.06%** | **0.9497** |

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
*   **`sbert_engine.py`**: Deep semantic understanding using fine-tuned **all-mpnet-base-v2** / **all-MiniLM-L12-v2** transformers.
*   **`tfidf_engine.py`**: Classic keyword-based matching (Lexical).
*   **`gensim_engine.py`**: Static word embeddings using **GloVe** (Wiki-Gigaword) and **Word2Vec** (Google News).
*   **`infersent_engine.py`**: Supervised sentence embeddings using a Bi-LSTM architecture.

### 3. **Linguistic Utilities (`utils/`)**
*   **`preprocessor.py`**: A bridge to an external **C-based NLP Preprocessor**. It extracts Tokens, POS tags, and NER entities with high performance.
*   **`nlp_helpers.py`**: Contains the scoring logic for structural overlap and the **Answer Synthesis** engine.
*   **`dataset.py`**: Handles the custom parsing of the multi-domain FAQ text file.

### 4. **Intent Prediction (`classifier/`)**
*   **`domain.py`**: Uses a **BERT-powered Logistic Regression** model. It extracts deep semantic embeddings for the query and predicts the target domain with **92.06% accuracy**. This is used to "boost" relevant results and penalize domain-mismatched answers.

### 5. **Configuration (`config.py`)**
The "Master Control Room." You can adjust system weights (Alpha, Beta, Gamma) and domain keywords here without editing the main logic.

---

## 🧪 The Hybrid Scoring Algorithm

The system does not rely on a single score. For every FAQ candidate, a **Final Confidence Score** is calculated using this formula:

$$FinalScore = [(\alpha \cdot Sim) + (\beta \cdot POS) + (\gamma \cdot NER)] \cdot DomainPenalty$$

*   **$\alpha$ (Similarity - 0.85)**: The raw semantic distance from the fine-tuned SBERT engine.
*   **$\beta$ (POS Match - 0.10)**: Measures how well the internal structure (nouns/verbs) matches.
*   **$\gamma$ (NER Overlap - 0.05)**: Ensures exact entity grounding.
*   **Domain Penalty (0.05x)**: If the system is 40%+ sure of a domain and a result is from another, the score is cut by 95% to prevent domain-leakage.

---

## 🧠 SBERT Fine-Tuning Workflow

To achieve the 92%+ accuracy jump, a specialized fine-tuning pipeline was implemented:

1.  **Data Augmentation (`fine_tuning/augment_data.py`)**: Uses a model-based approach to generate high-quality paraphrases for every FAQ in the dataset, creating a robust training set.
2.  **Specialized Training (`fine_tuning/train_sbert.py`)**: Fine-tunes the `SentenceTransformer` model using **MultipleNegativesRankingLoss** to map natural conversational queries to formal FAQ answers.
3.  **Checkpoint Management**: The best models are saved in `fine_tuned_sbert_v2/` for production inference.

---

## ✍️ Answer Synthesis (TL;DR)

The **Synthesized Summary** prepares a human-readable "Long Answer" by aggregating the Top 3 results:
*   **Single Domain**: Combines the most relevant context from the #1 result.
*   **Multi-Domain**: Collects relevant sentences from different domains (e.g., *"Regarding Medical: ... Regarding Legal: ..."*).

---

## 📊 Evaluation Metrics

When you run `/evaluate`, the system calculates professional IR metrics:
1.  **P@1 (Precision at 1)**: Accuracy of the top-ranked result.
2.  **MRR (Mean Reciprocal Rank)**: Quality of the overall ranking list.

---

## 🛠️ Replication & Training (For Collaborators)

If you have just cloned this repository, the system will use **pre-trained base models** by default (achieving ~76% accuracy). To replicate the professional-grade **92.06% accuracy**, you must locally train the SBERT engine:

### 1. Generate Training Data
Run the augmentation script to create a robust dataset of paraphrased query pairs:
```bash
python FAQ_Retrieval_System/fine_tuning/augment_data.py
```

### 2. Fine-Tune the SBERT Model
Train the transformer model on the augmented dataset:
```bash
python FAQ_Retrieval_System/fine_tuning/train_sbert.py
```
*The model will be saved to `FAQ_Retrieval_System/fine_tuned_sbert_v2/` and will be automatically detected by the core orchestrator on the next run.*

---

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execute the System**:
    ```bash
    python FAQ_Retrieval_System/faq_system.py
    ```
3.  **Commands**:
    *   `[Any Question]`: Standard search + synthesis.
    *   `/evaluate`: Run the Comparative Accuracy Benchmarks.
    *   `/exit`: Stop the program.
