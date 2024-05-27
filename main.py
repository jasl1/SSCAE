import json
from utils.lemmatizer import get_lemmatizer
from utils.text_processing import tokenize_text, detokenize_text, is_stop_word, remove_stop_words
from perturbation.perturbation_manager import Perturbation_Manager
from models.gpt2_wrapper import GPT2_wrapper
from models.gec import GEC
from models.st import ST
from models.use import USE
from models.infersent import INFERSENT
from models.sif_embedding import SIF_embedding
from perturbation.substitution_candidate_generation import Substitution_Candidate_Generation
from constraints.semantic_similarity_constraint import Semantic_Similarity_Constraint
from constraints.syntactic_similarity_constraint import Syntactic_Similarity_Constraint
from constraints.grammatical_correctness_constraint import Grammatical_Correctness_Constraint
from constraints.perturbation_constraints import Perturbation_Constraints

def main(text, gpt2_path, gec_path, st_path, use_path, infersent_model_path, infersent_vocab_path, sif_embedding_path, sif_freq_path, embedding_model_path, stop_words_path, top_n_percent, top_n_candidates, windows, device):
    gpt2 = GPT2_wrapper(gpt2_path)
    gec = GEC(gec_path)
    st = ST(st_path, windows, device)
    use = USE(use_path, windows, device)
    infersent = INFERSENT(infersent_model_path, infersent_vocab_path, windows, device)
    sif = SIF_embedding(sif_embedding_path, sif_freq_path)
    embedding_model = Substitution_Candidate_Generation(embedding_model_path, top_n_candidates)

    with open(stop_words_path, 'r') as f:
        stop_words = set(f.read().splitlines())

    semantic_constraint = Semantic_Similarity_Constraint(st, 0.9)
    syntactic_constraint = Syntactic_Similarity_Constraint(gpt2, 0.1)
    grammatical_constraint = Grammatical_Correctness_Constraint(gec, 1)
    constraints = Perturbation_Constraints([semantic_constraint, syntactic_constraint, grammatical_constraint])

    perturbation_manager = Perturbation_Manager(gpt2, gec, st, use, infersent, sif, embedding_model, top_n_candidates, constraints)

    lemmatizer = get_lemmatizer()
    perturbed_text = perturbation_manager.perturb(text, stop_words, lemmatizer, top_n_percent, top_n_candidates)
    print(perturbed_text)

if __name__ == "__main__":
    main(text="The quick brown fox jumps over the lazy dog.",
         gpt2_path="gpt2_model.pth",
         gec_path="gec_model.pth",
         st_path="st_model.pth",
         use_path="use_model.pth",
         infersent_model_path="infersent_model.pth",
         infersent_vocab_path="infersent_vocab.txt",
         sif_embedding_path="sif_embedding.txt",
         sif_freq_path="sif_freq.txt",
         embedding_model_path="embedding_model.pth",
         stop_words_path="stop_words.txt",
         top_n_percent=0.1,
         top_n_candidates=5,
         windows=[3, 5, 7],
         device="cuda")
