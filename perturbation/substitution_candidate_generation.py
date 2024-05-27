class Substitution_Candidate_Generation:
    def __init__(self, embedding_model, candidate_number):
        self.embedding_model = embedding_model
        self.candidate_number = candidate_number

    def generate_candidates(self, words, index):
        candidates = []
        target_word = words[index]
        for candidate in self.embedding_model.most_similar(target_word, topn=self.candidate_number):
            candidates.append(candidate[0])
        return candidates
