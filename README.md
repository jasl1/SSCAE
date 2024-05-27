# SSCAE
This project focuses on generating perturbed text from clean input text in a way that can potentially fool target models. By applying various constraints and perturbation techniques, the generated text aims to maintain semantic and syntactic coherence while altering its meaning.

### How It Works
The project consists of several modules responsible for different aspects of the perturbation process:

* Models: Contains wrappers and implementations of various language models used for text generation and analysis.
* Constraints: Implements constraints to ensure the perturbed text maintains certain characteristics, such as semantic similarity, syntactic structure, and grammatical correctness.
* Perturbation: Handles the generation and selection of perturbations based on candidate generation and selection algorithms.
* Utils: Provides utility functions for text processing and lemmatization.
