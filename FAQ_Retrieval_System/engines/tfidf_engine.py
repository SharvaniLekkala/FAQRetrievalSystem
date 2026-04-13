from .base import BaseEngine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFEngine(BaseEngine):
    def __init__(self, faqs):
        super().__init__(faqs)
        self.name = "TF-IDF"
        self.vectorizer = TfidfVectorizer(stop_words='english', tokenizer=lambda x: x.split(), lowercase=False, token_pattern=None)
        self.tfidf_matrix = None

    def train(self):
        texts = [f["tokens"] for f in self.faqs]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def get_similarity(self, query_tokens):
        query_vec = self.vectorizer.transform([query_tokens])
        return cosine_similarity(query_vec, self.tfidf_matrix).flatten()
