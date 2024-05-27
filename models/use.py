import tensorflow as tf
import tensorflow_hub as hub

class USE:
    def __init__(self, model_path, windows, device):
        self.model = hub.load(model_path)
        self.windows = windows
        self.device = device

    def get_sims_at_index(self, prev_words, candidates, index):
        # Your implementation
        pass
