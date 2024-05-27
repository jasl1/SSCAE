from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch

class YelpDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        return inputs, torch.tensor(label)

class Trainer:
    def __init__(self, model, tokenizer, train_data, test_data, batch_size=16, epochs=3, lr=5e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def train(self):
        train_dataset = YelpDataset(self.train_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        for epoch in range(self.epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}')

    def evaluate(self):
        test_dataset = YelpDataset(self.test_data, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        print(f'Accuracy: {accuracy}')
