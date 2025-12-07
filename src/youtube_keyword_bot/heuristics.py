from .text_processing import TextProcessing
from typing import List, Tuple


# Provides scoring, similarity evaluation, and keyword-pool construction for dataset comparison and GA initialization.
class Heuristics:
    # Creates a heuristics engine using a shared TextProcessing instance.
    def __init__(self, text_processor=None):
        self.tp = text_processor if text_processor else TextProcessing()

    # Scores every dataset entry against the input title and categories using weighted similarity metrics.
    def score_dataset(self, dataset, input_title, input_categories=None):
        input_tokens = self.tp.tokenize(input_title)
        input_vec = self.tp.vectorize(input_tokens)

        scored = []

        for entry in dataset:
            if input_title.strip().lower() == entry.title.strip().lower():
                scored.append((entry, 1.0))
                continue

            row_tokens = self.tp.tokenize(entry.title)
            row_vec = self.tp.vectorize(row_tokens)
            title_sim = self.tp.similarity(input_vec, row_vec)
            if title_sim < 0:
                title_sim = 0.0

            input_nouns = self._extract_nouns(input_title)
            row_nouns = self._extract_nouns(entry.title)

            if input_nouns and row_nouns:
                shared_nouns = len(set(input_nouns) & set(row_nouns))
                noun_overlap = shared_nouns / max(len(set(input_nouns)), 1)
            else:
                noun_overlap = 0.0

            input_title_words = set(
                w.lower() for w in input_title.replace("'", " ").split()
            )
            row_keywords = set(kw.lower() for kw in entry.keywords)

            shared_kw = len(input_title_words & row_keywords)
            keyword_overlap = shared_kw / max(len(input_title_words), 1)

            if input_categories:
                shared_cats = set(c.lower() for c in input_categories) & set(
                    c.lower() for c in entry.categories
                )
                category_match = len(shared_cats) / max(len(input_categories), 1)
            else:
                category_match = 0.0

            total_score = (
                0.40 * keyword_overlap +
                0.30 * noun_overlap +
                0.20 * category_match +
                0.10 * title_sim
            )

            total_score = max(0.0, min(1.0, total_score))

            scored.append((entry, total_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # Placeholder for future GA integration for keyword semantic bonus.
    def _keyword_relevance_bonus(self, entry_keywords, input_tokens):
        return 0.0

    # Filters raw keyword pools by comparing them to the input title to remove off-topic terms.
    def filter_pool_by_title(self, pool, input_title, sim_threshold=0.25):
        filtered = []

        doc_title = self.tp.nlp(input_title)
        title_tokens = {
            t.lemma_.lower()
            for t in doc_title
            if not t.is_punct and not t.is_space
        }

        for kw in pool:
            kw = kw.strip()
            if not kw:
                continue

            kw_doc = self.tp.nlp(kw)
            if not kw_doc:
                continue

            sim = 0.0
            if kw_doc[0].has_vector and doc_title.vector.any():
                sim = doc_title.similarity(kw_doc)
            if sim < 0:
                sim = 0.0

            kw_tokens = {
                t.lemma_.lower()
                for t in kw_doc
                if not t.is_punct and not t.is_space
            }
            overlap = bool(title_tokens & kw_tokens)

            if sim >= sim_threshold or overlap:
                filtered.append(kw)

        return filtered

    # Builds a combined keyword pool using dataset overlap and semantic expansions.
    def build_keyword_pool(self, scored_list, top_n=10, max_pool_size=200):
        freq = {}

        for entry, score in scored_list[:top_n]:
            for kw in entry.keywords:
                kw = kw.strip().lower()
                if kw:
                    freq[kw] = freq.get(kw, 0) + 1

        sorted_kws = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

        dataset_keywords = [kw for kw, _ in sorted_kws][:max_pool_size]

        title = scored_list[0][0].title
        title_nouns = self._extract_nouns(title)

        semantic_expanded = self._expand_nouns(title_nouns, limit=50)

        combined = dataset_keywords + semantic_expanded
        combined = list(dict.fromkeys(combined))

        return combined[:max_pool_size]

    # Extracts noun and proper-noun lemmas from a title.
    def _extract_nouns(self, text):
        doc = self.tp.nlp(text)
        nouns = [token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")]
        return nouns

    # Expands noun lemmas using embedding similarity to related terms.
    def _expand_nouns(self, nouns, limit=50):
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

    # Expands keywords using semantic neighborhoods for GA refinement.
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

    # Generates the final ranked list and keyword pool for use in the GA.
    def get_top_n_and_pool(self, dataset, input_title, input_categories=None, n=10):
        ranked = self.score_dataset(dataset, input_title, input_categories)
        pool = self.build_keyword_pool(ranked, top_n=n)
        return ranked[:n], pool
