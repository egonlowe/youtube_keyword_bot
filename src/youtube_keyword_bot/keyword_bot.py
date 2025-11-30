from typing import List, Optional, Dict

from .text_processing import TextProcessing
from .heuristics import Heuristics
from .genetic_algorithm import KeywordGA, GAConfig


class KeywordBot:
    """
    High-level pipeline for:
      title + categories
        -> heuristic similar-video search
        -> keyword pool building + filtering
        -> GA optimization
        -> final keyword list
    """

    def __init__(self, dataset, ga_config: Optional[GAConfig] = None):
        self.tp = TextProcessing()
        self.heur = Heuristics(self.tp)
        self.dataset = dataset
        self.ga_config = ga_config or GAConfig()

    def generate_keywords(
        self,
        input_title: str,
        input_categories: Optional[List[str]] = None,
        top_n_videos: int = 10,
        sim_threshold: float = 0.25,
    ) -> List[str]:
        """
        Main entrypoint:
        Given a title + optional categories, return an optimized keyword list
        using heuristics + GA.
        """

        # 1) Heuristics: rank videos and build raw keyword pool
        ranked, pool = self.heur.get_top_n_and_pool(
            self.dataset, input_title, input_categories, n=top_n_videos
        )

        # 2) Filter pool to keep only title-relevant keywords
        filtered_pool = self.heur.filter_pool_by_title(
            pool, input_title, sim_threshold=sim_threshold
        )

        if not filtered_pool:
            # Fallback: if filtering is too strict, just use raw pool
            filtered_pool = pool[:]

        # 3) Build frequency map from top-N videos (dataset-driven importance)
        freq: Dict[str, int] = {}
        for entry, score in ranked[:top_n_videos]:
            for kw in entry.keywords:
                kw_clean = kw.strip().lower()
                if kw_clean:
                    freq[kw_clean] = freq.get(kw_clean, 0) + 1

        max_freq = max(freq.values()) if freq else 1

        # Prepare NLP objects from title once
        title_doc = self.tp.nlp(input_title)
        title_tokens = {
            t.lemma_.lower()
            for t in title_doc
            if not t.is_punct and not t.is_space
        }

        # 4) Compute 3-way weights for each keyword in the filtered pool:
        #    (title lexical overlap + semantic similarity + dataset frequency) / 3
        weights: Dict[str, float] = {}

        for kw in filtered_pool:
            kw_str = kw.strip()
            if not kw_str:
                continue

            kw_doc = self.tp.nlp(kw_str)

            # (a) TITLE LEXICAL OVERLAP: 1.0 if any shared lemma, else 0.0
            kw_tokens = {
                t.lemma_.lower()
                for t in kw_doc
                if not t.is_punct and not t.is_space
            }
            has_overlap = bool(title_tokens & kw_tokens)
            overlap_score = 1.0 if has_overlap else 0.0

            # (b) SEMANTIC SIMILARITY: normalized to [0,1]
            sim = 0.0
            if kw_doc and kw_doc[0].has_vector and title_doc.vector.any():
                sim = title_doc.similarity(kw_doc)
            sim = max(0.0, min(sim, 1.0))  # clamp

            # (c) DATASET FREQUENCY: normalized by max_freq
            raw_freq = freq.get(kw_str.lower(), 0)
            freq_score = raw_freq / max_freq if max_freq > 0 else 0.0

            # Combine equally: title match, semantic sim, frequency/genre
            combined_score = (overlap_score + sim + freq_score) / 3.0

            # Small floor value so low-but-not-zero items aren’t invisible
            weights[kw_str] = max(combined_score, 0.05)

        # 5) Run GA on the filtered, weighted pool
        ga = KeywordGA(filtered_pool, keyword_weights=weights, config=self.ga_config)
        best = ga.run()

        # 6) Light cleanup: drop very sentence-like or known junk tags
        cleaned = [
            kw for kw in best
            if len(kw.split()) <= 4 and kw.lower() not in {"does anyone ready these"}
        ]

        # GA already respects char limit; cleanup always shortens, so we’re safe
        return cleaned
