import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths
PREPROCESSOR_EXE = os.path.join(BASE_DIR, "preprocessor.exe")
DATASET_FILE = os.path.join(BASE_DIR, "dataset.txt")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "test_ground_truth.json")

# Weights for multi-feature scoring
ALPHA = 0.5  # Semantic/Lexical Similarity
BETA = 0.25  # POS Match 
GAMMA = 0.25 # NER Entity Overlap

# Domain Logic constants
DOMAIN_KEYWORDS = {
    "medical": ["doctor", "medicine", "health", "symptoms", "treatment", "virus", "pain", "fever", "hospital", "burn"],
    "legal": ["law", "contract", "court", "legal", "felony", "misdemeanor", "attorney", "judge", "copyright", "trademark"],
    "tech": ["technology", "ssd", "software", "hardware", "cpu", "server", "cloud", "ai", "ml", "database", "api"],
    "finance": ["money", "budget", "stock", "market", "finance", "bank", "invest", "tax", "ira", "401k", "debt"],
    "education": ["learning", "school", "student", "teacher", "degree", "university", "study", "sat", "gre", "curriculum"]
}
