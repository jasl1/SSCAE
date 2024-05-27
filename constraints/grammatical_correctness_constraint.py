class Grammatical_Correctness_Constraint:
    def __init__(self, model, error_threshold):
        self.model = model
        self.error_threshold = error_threshold

    def check_constraint(self, words, candidates, index):
        return self.model.check_grammatical_constraint(words, candidates, index)
