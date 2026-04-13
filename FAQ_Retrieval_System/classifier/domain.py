import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class DomainClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', tokenizer=lambda x: x.split(), lowercase=True, token_pattern=None)
        self.model = LogisticRegression(max_iter=1000)
    
    def train(self, faqs):
        texts = [f["question"].lower() for f in faqs]
        y = [f["domain"] for f in faqs]
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, y)
    
    def predict(self, query):
        X = self.vectorizer.transform([query.lower()])
        probs = self.model.predict_proba(X)[0]
        max_idx = np.argmax(probs)
        return self.model.classes_[max_idx], probs[max_idx]
