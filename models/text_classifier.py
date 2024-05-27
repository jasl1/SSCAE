from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

class TextClassifier:
    def __init__(self, model_name: str):
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = AlbertForSequenceClassification.from_pretrained(model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions
