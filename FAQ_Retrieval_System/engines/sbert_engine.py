from .base import BaseEngine
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SBERTEngine(BaseEngine):
    def __init__(self, faqs, model_name='all-MiniLM-L6-v2'):
        super().__init__(faqs)
        self.name = f"SBERT ({model_name})"
        self.model = SentenceTransformer(model_name)
        self.faq_embeddings = None

    def train(self):
        print(f"Encoding {len(self.faqs)} FAQs with SBERT...")
        questions = [f["question"] for f in self.faqs]
        self.faq_embeddings = self.model.encode(questions, show_progress_bar=True)

    def get_similarity(self, query):
        query_emb = self.model.encode([query])
        return cosine_similarity(query_emb, self.faq_embeddings).flatten()
