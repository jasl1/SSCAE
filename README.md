# SSCAE
This project focuses on generating perturbed text from clean input text in a way that can potentially fool target models. By applying various constraints and perturbation techniques, the generated text aims to maintain semantic and syntactic coherence while altering its meaning.

### How It Works
The project consists of several modules responsible for different aspects of the perturbation process:

* **Models**: Contains wrappers and implementations of various language models used for text generation and analysis.
* **Constraints**: Implements constraints to ensure the perturbed text maintains certain characteristics, such as semantic similarity, syntactic structure, and grammatical correctness.
* **Perturbation**: Handles the generation and selection of perturbations based on candidate generation and selection algorithms.
* **Utils**: Provides utility functions for text processing and lemmatization.

### Generating Perturbed Text
1. **Setup**: Ensure all required models are available and paths are correctly set in main.py.
2. **Configuration**: Adjust file paths and parameters in main.py according to your requirements.
3. **Run**: Execute the script main.py to generate perturbed text from clean input text.

### How It Can Fool Target Models
The perturbed text generated by this project aims to maintain high similarity with the original text while fooling the target model. By doing so, it can potentially evade detection by target models that rely on semantic or syntactic patterns to make wrong predictions.

### Running the Script
1. Place your input text and required models in the appropriate directories.
2. Adjust the file paths and parameters in main.py as needed.
3. Run the script:
```bash
python main.py
```

### Contributing
Feel free to contribute to this project by submitting issues or pull requests. Ensure that your contributions align with the project's coding standards and include appropriate tests.
