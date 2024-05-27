class Perturbation_Constraints:
    def __init__(self, constraints):
        self.constraints = constraints

    def check_constraints(self, words, candidates, index):
        results = []
        for constraint in self.constraints:
            results.append(constraint.check_constraint(words, candidates, index))
        return np.all(results, axis=0)
