# 🚀 Multi-Engine FAQ System: Technical Documentation

A high-performance, modular NLP suite designed for high-accuracy FAQ retrieval across multiple domains (Medical, Legal, Finance, Tech, Education). The system utilizes a hybrid scoring mechanism combining Transformer-based semantics, lexical statistics, and linguistic structure (POS/NER).

---

## 🏁 Performance Breakthrough: 69% → 92%

The system has undergone a major optimization journey to reach professional-grade accuracy (**92.06% P@1**) on paraphrased user queries.

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
A swappable architecture where every engine inherits from a common `BaseEngine`:
*   **`sbert_engine.py`**: Deep semantic understanding using fine-tuned **all-mpnet-base-v2** / **all-MiniLM-L12-v2** transformers.
*   **`tfidf_engine.py`**: Classic keyword-based matching (Lexical).
*   **`gensim_engine.py`**: Static word embeddings using **GloVe** (Wiki-Gigaword) and **Word2Vec** (Google News).
*   **`infersent_engine.py`**: Supervised sentence embeddings using a Bi-LSTM architecture.

### 3. **Linguistic Utilities (`utils/`)**
*   **`preprocessor.py`**: A bridge to an external **C-based NLP Preprocessor**. It extracts Tokens, POS tags, and NER entities with high performance.
*   **`nlp_helpers.py`**: Contains the scoring logic for structural overlap and the **Answer Synthesis** engine.
*   **`dataset.py`**: Handles the custom parsing of the multi-domain FAQ text file.

### 4. **Intent Prediction (`classifier/`)**
*   **`domain.py`**: Uses a **BERT-powered Logistic Regression** model. It predicts the target domain (Medical, Tech, etc.) with **92.06% accuracy**, applying a 95% penalty to domain-mismatched results.

---

## 🧩 Technological Stack
*   **Core**: Python 3.10+
*   **Deep Learning**: PyTorch, Sentence-Transformers (BERT/MPNet)
*   **Machine Learning**: Scikit-learn (Logistic Regression, Cosine Similarity)
*   **Natural Language Processing**: Gensim (GloVe/Word2Vec), C-based Custom Tokenizer
*   **Performance**: C (Binary Preprocessor)

---

## 🧪 The Hybrid Scoring Algorithm

For every FAQ candidate, a **Final Confidence Score** is calculated:

$$FinalScore = [(\alpha \cdot Sim) + (\beta \cdot POS) + (\gamma \cdot NER)] \cdot DomainPenalty$$

*   **$\alpha$ (Similarity - 0.85)**: Raw semantic distance from the fine-tuned SBERT engine.
*   **$\beta$ (POS Match - 0.10)**: Structural matching of nouns and verbs.
*   **$\gamma$ (NER Overlap - 0.05)**: Named Entity grounding.
*   **Domain Penalty (0.05x)**: Penalizes results outside the predicted query intent.

---

## 🧠 SBERT Fine-Tuning Workflow

To replicate the 92%+ accuracy locally:

1.  **Data Augmentation (`fine_tuning/augment_data.py`)**: Generates high-quality paraphrases for training.
2.  **Specialized Training (`fine_tuning/train_sbert.py`)**: Fine-tunes the transformer using **MultipleNegativesRankingLoss**.

---

## 🛠️ Installation & Setup (For Collaborators)

### 1. Prerequisites
Ensure you have the Python dependencies installed:
```bash
pip install -r FAQ_Retrieval_System/requirements.txt
```

### 2. C Preprocessor (Optional / Cross-Platform)
The repository includes `preprocessor.exe` for Windows. To recompile for Linux/Mac:
```bash
gcc FAQ_Retrieval_System/preprocessor.c -o FAQ_Retrieval_System/preprocessor.exe
```

### 3. Run the System
```bash
python FAQ_Retrieval_System/faq_system.py
```

---

## 📁 Repository Structure
```text
.
├── FAQ_Retrieval_System/
│   ├── classifier/         # BERT-based Domain Intent classification
│   ├── engines/            # Lexical and Semantic embedding engines
│   ├── fine_tuning/        # Training scripts for model specialization
│   ├── scratch/            # Optimization and research scripts
│   ├── utils/              # NLP structure and dataset helpers
│   ├── dataset.txt         # Core FAQ data (Multi-domain)
│   ├── config.py           # Global weights and directory paths
│   └── faq_system.py       # Main Application Entry Point
├── README.md               # Unified Technical Documentation
└── .gitignore              # Repository safety and large file handling
```
