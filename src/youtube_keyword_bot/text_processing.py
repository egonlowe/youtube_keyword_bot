import spacy
import numpy as np

class TextProcessing:

    # Initializes the spaCy language model used for tokenization and vector similarity.
    def __init__(self):
        # Load small English model with word vectors
        self.nlp = spacy.load('en_core_web_md') # replace with _sm if it runs poorly

    # Lowercases text and removes punctuation to prepare it for tokenization.
    def normalize(self, text):
        # Lowercase, strip whitespace
        return text.lower().strip()

    # Converts text into filtered tokens (no stopwords, no punctuation).
    def tokenize(self, text):
        doc = self.nlp(text)
        return [token.lemma_.lower() for token in doc
                if not token.is_stop and not token.is_punct] # touch up logic for this since it still takes in "|"

    # Averages the word vectors of all tokens to create a single vector representation.
    def vectorize(self, tokens):
        # average word embeddings
        vectors = []
        for token in tokens:
            if token in self.nlp.vocab and self.nlp.vocab[token].has_vector:
                vectors.append(self.nlp.vocab[token].vector)

        if not vectors:
            return np.zeros((300,)) # fallback vector

        return np.mean(np.array(vectors), axis=0)

    # Computes cosine similarity between two vector representations.
    def similarity(self, vec1, vec2):
        # Cosine similarity
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
