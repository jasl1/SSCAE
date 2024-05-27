import pandas as pd
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        train_data = pd.DataFrame(dataset['train'])
        test_data = pd.DataFrame(dataset['test'])
        return train_data, test_data
