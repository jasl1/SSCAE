from sentence_transformers import SentenceTransformer

class ST:
    def __init__(self, model_path, windows, device):
        self.model = SentenceTransformer(model_path)
        self.windows = windows
        self.device = device

    def get_sims_at_index(self, prev_words, candidates, index):
        # Your implementation
        pass
