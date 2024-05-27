from nltk.tokenize import word_tokenize, TreebankWordDetokenizer

def tokenize_text(text):
    return word_tokenize(text)

def detokenize_text(words):
    return TreebankWordDetokenizer().detokenize(words)

def is_stop_word(word, stop_words):
    return word.lower() in stop_words

def remove_stop_words(words, stop_words):
    return [word for word in words if not is_stop_word(word, stop_words)]
