import numpy as np

class Perturbation_Selection:
    def __init__(self, pert_order, pert_direction):
        self.pert_order = pert_order
        self.pert_direction = pert_direction

    def select_top_n(self, pert_scores, N):
        if self.pert_order == "ascending":
            return np.argsort(pert_scores)[:N]
        else:
            return np.argsort(pert_scores)[::-1][:N]

    def filter_by_threshold(self, pert_scores, threshold):
        if self.pert_direction == "above":
            return [i for i, score in enumerate(pert_scores) if score >= threshold]
        else:
            return [i for i, score in enumerate(pert_scores) if score < threshold]

    def select_top_n_percent(self, pert_scores, percent):
        N = int(len(pert_scores) * percent)
        return self.select_top_n(pert_scores, N)
