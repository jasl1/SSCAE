from nltk.stem import WordNetLemmatizer

def get_lemmatizer():
    return WordNetLemmatizer()

def lemmatize_words(words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in words]

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'
