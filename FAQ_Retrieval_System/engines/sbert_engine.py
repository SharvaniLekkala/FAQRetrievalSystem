import os
from .base import BaseEngine
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import FINETUNED_SBERT_PATH

class SBERTEngine(BaseEngine):
    def __init__(self, faqs, model_name='all-mpnet-base-v2'):
        super().__init__(faqs)
        
        # Prefer fine-tuned model if it exists
        if os.path.exists(FINETUNED_SBERT_PATH):
            print(f"Loading local fine-tuned SBERT model from: {FINETUNED_SBERT_PATH}")
            self.model = SentenceTransformer(FINETUNED_SBERT_PATH)
            self.name = "SBERT (Fine-tuned)"
        else:
            print(f"Loading pre-trained SBERT model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.name = f"SBERT ({model_name})"
            
        self.faq_embeddings = None

    def train(self):
        print(f"Encoding {len(self.faqs)} FAQs with SBERT...")
        questions = [f["question"] for f in self.faqs]
        self.faq_embeddings = self.model.encode(questions, show_progress_bar=True)

    def get_similarity(self, query):
        query_emb = self.model.encode([query])
        return cosine_similarity(query_emb, self.faq_embeddings).flatten()
