from .substitution_candidate_generation.py import Substitution_Candidate_Generation
from .perturbation_selection.py import Perturbation_Selection

class Perturbation_Manager:
    def __init__(self, gpt2, gec, st, use, infersent, sif, embedding_model, candidate_number, constraints):
        self.gpt2 = gpt2
        self.gec = gec
        self.st = st
        self.use = use
        self.infersent = infersent
        self.sif = sif
        self.embedding_model = embedding_model
        self.candidate_number = candidate_number
        self.constraints = constraints

    def generate_candidates(self, words, index):
        return self.embedding_model.generate_candidates(words, index)

    def check_constraints(self, words, candidates, index):
        return self.constraints.check_constraints(words, candidates, index)

    def select_top_candidates(self, words, candidates, index, top_n_percent):
        scores = self.st.get_sims_at_index(words, candidates, index)
        top_indices = Perturbation_Selection("descending", "above").select_top_n_percent(scores, top_n_percent)
        return [candidates[i] for i in top_indices]

    def perturb(self, text, stop_words, lemmatizer, top_n_percent, top_n_candidates):
        words = tokenize_text(text)
        for index in range(len(words)):
            if is_stop_word(words[index], stop_words):
                continue
            candidates = self.generate_candidates(words, index)
            candidates = self.check_constraints(words, candidates, index)
            top_candidates = self.select_top_candidates(words, candidates, index, top_n_percent)
            if top_candidates:
                words[index] = top_candidates[0]
        return detokenize_text(words)
