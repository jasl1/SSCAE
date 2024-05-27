class Syntactic_Similarity_Constraint:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def check_constraint(self, words, candidates, index):
        diffs = self.model.get_probs_diff_at_index(words, candidates, index)
        return [diff <= self.threshold for diff in diffs]
