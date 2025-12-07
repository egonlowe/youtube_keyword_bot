from typing import List, Optional, Dict

from .text_processing import TextProcessing
from .heuristics import Heuristics
from .genetic_algorithm import KeywordGA, GAConfig


# High-level controller that runs the full pipeline: heuristics, filtering, weighting, and genetic optimization.
class KeywordBot:
    # Creates a KeywordBot with text processing, heuristics engine, dataset, and GA configuration.
    def __init__(self, dataset, ga_config: Optional[GAConfig] = None):
        self.tp = TextProcessing()
        self.heur = Heuristics(self.tp)
        self.dataset = dataset
        self.ga_config = ga_config or GAConfig()

    # Generates a final keyword list using heuristics, filtering, weighted scoring, and GA optimization.
    def generate_keywords(
        self,
        input_title: str,
        input_categories: Optional[List[str]] = None,
        top_n_videos: int = 10,
        sim_threshold: float = 0.25,
    ) -> List[str]:

        ranked, pool = self.heur.get_top_n_and_pool(
            self.dataset, input_title, input_categories, n=top_n_videos
        )

        filtered_pool = self.heur.filter_pool_by_title(
            pool, input_title, sim_threshold=sim_threshold
        )

        if not filtered_pool:
            filtered_pool = pool[:]

        freq: Dict[str, int] = {}
        for entry, score in ranked[:top_n_videos]:
            for kw in entry.keywords:
                kw_clean = kw.strip().lower()
                if kw_clean:
                    freq[kw_clean] = freq.get(kw_clean, 0) + 1

        max_freq = max(freq.values()) if freq else 1

        title_doc = self.tp.nlp(input_title)
        title_tokens = {
            t.lemma_.lower()
            for t in title_doc
            if not t.is_punct and not t.is_space
        }

        weights: Dict[str, float] = {}

        for kw in filtered_pool:
            kw_str = kw.strip()
            if not kw_str:
                continue

            kw_doc = self.tp.nlp(kw_str)

            kw_tokens = {
                t.lemma_.lower()
                for t in kw_doc
                if not t.is_punct and not t.is_space
            }
            has_overlap = bool(title_tokens & kw_tokens)
            overlap_score = 1.0 if has_overlap else 0.0

            sim = 0.0
            if kw_doc and kw_doc[0].has_vector and title_doc.vector.any():
                sim = title_doc.similarity(kw_doc)
            sim = max(0.0, min(sim, 1.0))

            raw_freq = freq.get(kw_str.lower(), 0)
            freq_score = raw_freq / max_freq if max_freq > 0 else 0.0

            combined_score = (overlap_score + sim + freq_score) / 3.0

            weights[kw_str] = max(combined_score, 0.05)

        ga = KeywordGA(filtered_pool, keyword_weights=weights, config=self.ga_config)
        best = ga.run()

        cleaned = [
            kw for kw in best
            if len(kw.split()) <= 4 and kw.lower() not in {"does anyone ready these"}
        ]

        return cleaned
