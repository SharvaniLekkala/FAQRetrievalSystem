from .base import BaseEngine
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GensimEngine(BaseEngine):
    """Engine for GloVe and Word2Vec."""
    def __init__(self, faqs, model_type='glove'):
        super().__init__(faqs)
        self.name = model_type.upper()
        
        model_map = {
            'glove': 'glove-wiki-gigaword-100',
            'word2vec': 'glove-twitter-50'
        }
        print(f"Loading {model_type} model ({model_map[model_type]})... this may take a while.")
        self.model = api.load(model_map[model_type])
        self.faq_embeddings = None

    def _get_sentence_vector(self, tokens_list):
        vectors = [self.model[word] for word in tokens_list if word in self.model]
        if not vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)

    def train(self):
        print(f"Encoding {len(self.faqs)} FAQs with {self.name}...")
        self.faq_embeddings = np.array([self._get_sentence_vector(f["tokens"].split()) for f in self.faqs])

    def get_similarity(self, query_tokens):
        query_vec = self._get_sentence_vector(query_tokens.split()).reshape(1, -1)
        return cosine_similarity(query_vec, self.faq_embeddings).flatten()
