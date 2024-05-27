import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2_wrapper:
    def __init__(self, model_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model.eval()

    def get_probs_diff_at_index(self, prev_words, candidates, index):
        # Your implementation
        pass
