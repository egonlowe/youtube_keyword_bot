import spacy
import numpy as np
import hashlib


class TextProcessing:

    def __init__(self):
        # Best balance of quality and speed
        self.nlp = spacy.load("en_core_web_md")
        self.vector_size = self.nlp.vocab.vectors.shape[1]

    # -------------------------------------------------------------
    # TOKENIZATION — ALWAYS RETURN CLEAN STRING TOKENS
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # SUBWORD FALLBACK VECTORS FOR OOV TERMS
    # -------------------------------------------------------------
    def _fallback_vector(self, token: str):
        """
        Creates a deterministic pseudo-vector for OOV tokens.
        Expands/repeats the MD5 hex string so slicing never runs out.
        """

        # Hash token into hex
        h = hashlib.md5(token.encode("utf-8")).hexdigest()  # 32 chars

        # Repeat hash until long enough
        needed = self.vector_size * 2  # number of hex chars needed
        extended = (h * ((needed // len(h)) + 1))[:needed]

        # Convert hex → numbers
        vec = np.array(
            [int(extended[i:i + 2], 16) for i in range(0, needed, 2)],
            dtype=float
        )

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    # -------------------------------------------------------------
    # VECTORIZATION — CONSISTENT, OOV-RESILIENT
    # -------------------------------------------------------------
    def vectorize(self, tokens):
        """
        tokens = list[str]
        Uses:
            - real spaCy vectors when available
            - subword fallback vectors when OOV
            - noun weighting via POS detection
        """

        if not tokens:
            return np.zeros((self.vector_size,))

        # Recreate a proper doc to get POS tags
        doc = self.nlp(" ".join(tokens))

        vectors = []

        for token in doc:
            text = token.text.lower()

            # Use spaCy vector if available
            if token.has_vector:
                vec = token.vector
            else:
                # Fallback vector for OOV tokens
                vec = self._fallback_vector(text)

            # Noun / Proper Noun weighting
            if token.pos_ in ("NOUN", "PROPN"):
                vec = vec * 1.15

            vectors.append(vec)

        if not vectors:
            return np.zeros((self.vector_size,))

        return np.mean(np.array(vectors), axis=0)

    # -------------------------------------------------------------
    # COSINE SIMILARITY
    # -------------------------------------------------------------
    def similarity(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
