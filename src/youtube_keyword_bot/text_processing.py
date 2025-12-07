import spacy
import numpy as np
import hashlib


# Handles all text normalization, tokenization, vectorization, and similarity scoring for the project.
class TextProcessing:

    # Initializes the spaCy model and determines vector size for embeddings.
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.vector_size = self.nlp.vocab.vectors.shape[1]

    # Tokenizes text, removes stopwords and punctuation, and ensures all returned tokens are usable strings.
    def tokenize(self, text):
        doc = self.nlp(text)

        tokens = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            if not any(ch.isalnum() for ch in token.text):
                continue

            tokens.append(token.lemma_.lower())

        return tokens

    # Generates a fallback vector for out-of-vocabulary words using a deterministic hashing strategy.
    def _fallback_vector(self, token: str):
        h = hashlib.md5(token.encode("utf-8")).hexdigest()

        needed = self.vector_size * 2
        extended = (h * ((needed // len(h)) + 1))[:needed]

        vec = np.array(
            [int(extended[i:i + 2], 16) for i in range(0, needed, 2)],
            dtype=float
        )

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    # Converts a list of tokens into an averaged vector embedding, handling OOV words and applying noun weighting.
    def vectorize(self, tokens):
        if not tokens:
            return np.zeros((self.vector_size,))

        doc = self.nlp(" ".join(tokens))

        vectors = []

        for token in doc:
            text = token.text.lower()

            if token.has_vector:
                vec = token.vector
            else:
                vec = self._fallback_vector(text)

            if token.pos_ in ("NOUN", "PROPN"):
                vec = vec * 1.15

            vectors.append(vec)

        if not vectors:
            return np.zeros((self.vector_size,))

        return np.mean(np.array(vectors), axis=0)

    # Computes cosine similarity between two vector embeddings.
    def similarity(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
