class BaseEngine:
    def __init__(self, faqs):
        self.faqs = faqs
        self.name = "Base"

    def train(self):
        pass

    def get_similarity(self, query):
        """Returns a list of scores for all FAQs."""
        raise NotImplementedError
