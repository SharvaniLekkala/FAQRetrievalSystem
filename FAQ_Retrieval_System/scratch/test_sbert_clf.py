import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Local imports
sys.path.append(os.getcwd())
from utils.dataset import parse_dataset
from config import DATASET_FILE, FINETUNED_SBERT_PATH

def train_sbert_classifier():
    print("Loading dataset...")
    faqs = parse_dataset(DATASET_FILE)
    
    texts = [f["question"] for f in faqs]
    y = [f["domain"] for f in faqs]
    
    print("Loading SBERT to extract features...")
    model = SentenceTransformer(FINETUNED_SBERT_PATH if os.path.exists(FINETUNED_SBERT_PATH) else 'all-mpnet-base-v2')
    
    print("Encoding questions...")
    X = model.encode(texts, show_progress_bar=True)
    
    print("Training Logistic Regression on top of SBERT embeddings...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    train_acc = clf.score(X, y) * 100
    print(f"Training Accuracy: {train_acc:.2f}%")
    
    return clf, model

if __name__ == "__main__":
    train_sbert_classifier()
