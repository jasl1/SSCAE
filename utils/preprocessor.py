import string

class Preprocessor:
    def __init__(self, stop_words_path: str):
        with open(stop_words_path, 'r') as file:
            self.stop_words = set(line.strip() for line in file)

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
