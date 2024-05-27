class Semantic_Similarity_Constraint:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def check_constraint(self, words, candidates, index):
        sims = self.model.get_sims_at_index(words, candidates, index)
        return [sim >= self.threshold for sim in sims]
