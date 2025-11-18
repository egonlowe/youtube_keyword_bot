from .text_processing import TextProcessing
from typing import List, Tuple


class Heuristics:
    def __init__(self, text_processor=None):
        # Shared TextProcessing instance (efficient)
        self.tp = text_processor if text_processor else TextProcessing()

    # ---------------------------------------------------------
    # SCORING DATASET ENTRIES
    # ---------------------------------------------------------
    def score_dataset(self, dataset, input_title, input_categories=None):

        """
        Scores each VideoEntry in the dataset using a dataset-first approach:
        • Keyword overlap (strongest)
        • Noun overlap (strong)
        • Category match (medium)
        • Semantic similarity (weakest)
        """

        input_tokens = self.tp.tokenize(input_title)
        input_vec = self.tp.vectorize(input_tokens)

        scored = []

        for entry in dataset:

            # =====================================================
            # PERFECT TITLE MATCH OVERRIDE
            # =====================================================
            if input_title.strip().lower() == entry.title.strip().lower():
                scored.append((entry, 1.0))
                continue

            # ===============================================
            # 1. TITLE SEMANTIC SIMILARITY (weak weight)
            # ===============================================
            row_tokens = self.tp.tokenize(entry.title)
            row_vec = self.tp.vectorize(row_tokens)
            title_sim = self.tp.similarity(input_vec, row_vec)
            if title_sim < 0:
                title_sim = 0.0

            # ===============================================
            # 2. NOUN OVERLAP (strong)
            # ===============================================
            input_nouns = self._extract_nouns(input_title)
            row_nouns = self._extract_nouns(entry.title)

            if input_nouns and row_nouns:
                shared_nouns = len(set(input_nouns) & set(row_nouns))
                noun_overlap = shared_nouns / max(len(set(input_nouns)), 1)
            else:
                noun_overlap = 0.0

            # ===============================================
            # 3. KEYWORD OVERLAP (strongest)
            # ===============================================
            # tokenize input title into basic lowercase words
            input_title_words = set(
                w.lower() for w in input_title.replace("'", " ").split()
            )
            row_keywords = set(kw.lower() for kw in entry.keywords)

            shared_kw = len(input_title_words & row_keywords)
            keyword_overlap = shared_kw / max(len(input_title_words), 1)

            # ===============================================
            # 4. CATEGORY MATCH (medium)
            # ===============================================
            if input_categories:
                shared_cats = set(c.lower() for c in input_categories) & set(
                    c.lower() for c in entry.categories
                )
                category_match = len(shared_cats) / max(len(input_categories), 1)
            else:
                category_match = 0.0

            # ===============================================
            # 5. FINAL WEIGHTED SCORE (normalized)
            # ===============================================
            total_score = (
                0.40 * keyword_overlap +     # strongest
                0.30 * noun_overlap +        # strong
                0.20 * category_match +      # medium
                0.10 * title_sim             # weakest
            )

            total_score = max(0.0, min(1.0, total_score))  # clamp 0–1

            scored.append((entry, total_score))

        # Highest score first
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ---------------------------------------------------------
    # KEYWORD RELEVANCE BONUS (unused now, but left for GA later)
    # ---------------------------------------------------------
    def _keyword_relevance_bonus(self, entry_keywords, input_tokens):
        """(Kept for future GA integration — no longer used for scoring)"""
        return 0.0

    # ---------------------------------------------------------
    # BUILD KEYWORD POOL
    # ---------------------------------------------------------
    def build_keyword_pool(self, scored_list, top_n=10, max_pool_size=200):
        """
        Build a keyword pool where:
        1) Primary keywords come from dataset videos
        2) Remaining slots are filled with semantic expansions of nouns from the input title
        """
        freq = {}

        # ------------------------------
        # PART 1 — DATASET KEYWORDS
        # ------------------------------
        for entry, score in scored_list[:top_n]:
            for kw in entry.keywords:
                kw = kw.strip().lower()
                if kw:
                    freq[kw] = freq.get(kw, 0) + 1

        # Sort by frequency
        sorted_kws = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

        dataset_keywords = [kw for kw, _ in sorted_kws][:max_pool_size]

        # ------------------------------
        # PART 2 — Title noun expansions
        # ------------------------------
        title = scored_list[0][0].title
        title_nouns = self._extract_nouns(title)

        semantic_expanded = self._expand_nouns(title_nouns, limit=50)

        combined = dataset_keywords + semantic_expanded
        combined = list(dict.fromkeys(combined))  # dedupe

        return combined[:max_pool_size]

    # ---------------------------------------------------------
    # NOUN EXTRACTION
    # ---------------------------------------------------------
    def _extract_nouns(self, text):
        """Extracts only nouns and proper nouns from the input title."""
        doc = self.tp.nlp(text)
        nouns = [token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")]
        return nouns

    # ---------------------------------------------------------
    # SEMANTIC EXPANSION OF NOUNS
    # ---------------------------------------------------------
    def _expand_nouns(self, nouns, limit=50):
        """Generate semantic expansions only from the title’s nouns."""
        expanded = set()

        for noun in nouns:
            try:
                doc = self.tp.nlp(noun)
                if not doc or not doc[0].has_vector:
                    continue

                similar = self.tp.nlp.vocab.vectors.most_similar(
                    doc[0].vector.reshape(1, -1), n=20
                )
                keys = similar[0][0]

                for key in keys:
                    word = self.tp.nlp.vocab.strings[key]
                    if word.isalpha() and 3 <= len(word) <= 12:
                        expanded.add(word.lower())

            except Exception:
                continue

        return list(expanded)[:limit]

    # ---------------------------------------------------------
    # SEMANTIC KEYWORD EXPANSION (GA USE LATER)
    # ---------------------------------------------------------
    def _expand_semantic_keywords(self, keywords, top_k=5):
        new_terms = set()

        for kw in keywords:
            doc = self.tp.nlp(str(kw))
            if not doc or not doc[0].has_vector:
                continue

            token = doc[0]
            try:
                similar = self.tp.nlp.vocab.vectors.most_similar(
                    token.vector.reshape(1, -1), n=top_k
                )
            except KeyError:
                continue

            keys = similar[0][0]

            for key in keys:
                word = self.tp.nlp.vocab.strings[key]
                if not word:
                    continue

                if len(word) < 3:
                    continue
                if not word[0].isalpha():
                    continue

                wdoc = self.tp.nlp(word)
                if not wdoc:
                    continue

                if wdoc[0].pos_ not in ("NOUN", "PROPN", "ADJ"):
                    continue

                new_terms.add(word.lower())

        return list(new_terms)

    # ---------------------------------------------------------
    # COMBINED ENTRYPOINT
    # ---------------------------------------------------------
    def get_top_n_and_pool(self, dataset, input_title, input_categories=None, n=10):
        ranked = self.score_dataset(dataset, input_title, input_categories)
        pool = self.build_keyword_pool(ranked, top_n=n)
        return ranked[:n], pool
