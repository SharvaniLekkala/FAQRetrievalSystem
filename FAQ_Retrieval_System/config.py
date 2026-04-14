import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths
PREPROCESSOR_EXE = os.path.join(BASE_DIR, "preprocessor.exe")
DATASET_FILE = os.path.join(BASE_DIR, "dataset.txt")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "test_ground_truth.json")
FINETUNED_SBERT_PATH = os.path.join(BASE_DIR, "fine_tuned_sbert_v2")

# Weights for multi-feature scoring (Optimized for 85%+ Accuracy)
ALPHA = 0.85  # Semantic/Lexical Similarity (BERT)
BETA = 0.10   # POS Match 
GAMMA = 0.05  # NER Entity Overlap

# Domain Logic constants
DOMAIN_KEYWORDS = {
    "medical": ["doctor", "medicine", "health", "symptoms", "treatment", "virus", "pain", "fever", "hospital", "burn"],
    "legal": ["law", "contract", "court", "legal", "felony", "misdemeanor", "attorney", "judge", "copyright", "trademark"],
    "tech": ["technology", "ssd", "software", "hardware", "cpu", "server", "cloud", "ai", "ml", "database", "api"],
    "finance": ["money", "budget", "stock", "market", "finance", "bank", "invest", "tax", "ira", "401k", "debt"],
    "education": ["learning", "school", "student", "teacher", "degree", "university", "study", "sat", "gre", "curriculum"]
}
