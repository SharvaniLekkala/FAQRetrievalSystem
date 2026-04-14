import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

class DomainClassifier:
    def __init__(self, model_name_or_path='all-mpnet-base-v2'):
        # We'll use the pre-trained or fine-tuned model for features
        self.model_path = model_name_or_path
        self._sbert = None
        self.model = LogisticRegression(max_iter=1000)
        self.classes_ = None

    @property
    def sbert(self):
        if self._sbert is None:
            print(f"Initializing Domain Classifier SBERT ({self.model_path})...")
            self._sbert = SentenceTransformer(self.model_path)
        return self._sbert

    def train(self, faqs):
        texts = [f["question"] for f in faqs]
        y = [f["domain"] for f in faqs]
        
        print("Extracting BERT features for domain classification...")
        X = self.sbert.encode(texts, show_progress_bar=True)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        print(f"Domain Classifier trained on {len(texts)} samples.")

    def predict(self, query):
        X = self.sbert.encode([query])
        probs = self.model.predict_proba(X)[0]
        max_idx = np.argmax(probs)
        return self.model.classes_[max_idx], probs[max_idx]
