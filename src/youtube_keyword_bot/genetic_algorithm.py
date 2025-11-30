

import random
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class GAConfig:
    population_size: int = 40
    min_keywords: int = 20
    max_keywords: int = 30
    max_chars: int = 500
    generations: int = 25
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 4


class KeywordGA:
    def __init__(
        self,
        keyword_pool: List[str],
        keyword_weights: Optional[Dict[str, float]] = None,
        config: Optional[GAConfig] = None,
        rng_seed: Optional[int] = None
    ):
        """
        keyword_pool: list of candidate keywords (from heuristics)
        keyword_weights: optional mapping of keyword -> importance (e.g., frequency)
        """
        # Dedup while preserving order
        self.pool = list(dict.fromkeys(keyword_pool))
        self.weights = keyword_weights or {kw: 1.0 for kw in self.pool}
        self.config = config or GAConfig()
        self.rng = random.Random(rng_seed)

    # ------------ Utility helpers ------------

    def _keywords_to_string(self, keywords: List[str]) -> str:
        """Format keywords as comma-separated for counting."""
        return ",".join(keywords)

    def _char_count(self, keywords: List[str]) -> int:
        """Return total character count of the keyword list when joined."""
        return len(self._keywords_to_string(keywords))

    def _within_limits(self, keywords: List[str]) -> bool:
        """Check keyword count & character budget constraints."""
        if not (self.config.min_keywords <= len(keywords) <= self.config.max_keywords):
            return False
        if self._char_count(keywords) > self.config.max_chars:
            return False
        return True

    # ------------ Individual creation ------------

    def make_random_individual(self) -> List[str]:
        """
        Create a random candidate keyword list:
        - Sample between min_keywords and max_keywords from pool
        - Enforce max_chars budget by skipping ones that overflow
        """
        if not self.pool:
            return []

        target_size = self.rng.randint(self.config.min_keywords, self.config.max_keywords)
        shuffled = self.pool[:]
        self.rng.shuffle(shuffled)

        individual: List[str] = []
        for kw in shuffled:
            candidate = individual + [kw]
            # respect keyword count upper bound
            if len(candidate) > target_size:
                break
            # respect char budget
            if self._char_count(candidate) > self.config.max_chars:
                continue
            individual = candidate

        return individual

    def make_initial_population(self) -> List[List[str]]:
        """Create the initial population of individuals."""
        population = []
        attempts = 0
        while len(population) < self.config.population_size and attempts < self.config.population_size * 5:
            indiv = self.make_random_individual()
            if indiv and self._within_limits(indiv):
                population.append(indiv)
            attempts += 1

            # Fallback: if pool or constraints are too tight, relax min_keywords a bit
            if len(population) == 0 and attempts > self.config.population_size * 2:
                self.config.min_keywords = max(5, self.config.min_keywords - 1)

        return population

    # ------------ Fitness ------------

    def fitness(self, individual: List[str]) -> float:
        """
        Fitness definition (v1):
        - Sum of keyword weights (from frequency or uniform)
        - Hard zero if > char budget
        - Bonus for using more of the char budget (but staying <= 500)
        - Bonus for having closer to max_keywords
        """
        if not individual:
            return 0.0

        total_chars = self._char_count(individual)
        if total_chars > self.config.max_chars:
            return 0.0

        # Base score from weights
        weight_score = sum(self.weights.get(kw, 1.0) for kw in individual)

        # Penalize very long, sentence-y keywords a bit
        junk_penalty = 0.0
        for kw in individual:
            # crude heuristic: more than 4 words = likely junky tag
            if len(kw.split()) > 4:
                junk_penalty += 0.5

        weight_score = max(0.0, weight_score - junk_penalty)

        # How well we use the character budget (0–1)
        budget_usage = total_chars / self.config.max_chars

        # How close we are to max_keywords (0–1)
        size_score = len(individual) / self.config.max_keywords

        # Combine components
        # You can tune these weights later if desired
        base = weight_score
        bonus = 0.3 * budget_usage + 0.2 * size_score  # 0–0.5 range-ish

        return base * (1.0 + bonus)

    # ------------ Selection (tournament) ------------

    def _tournament_select(self, population: List[List[str]], fitnesses: List[float]) -> List[str]:
        """Tournament selection for one parent."""
        indices = [self.rng.randrange(len(population)) for _ in range(self.config.tournament_size)]
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx][:]  # copy

    # ------------ Crossover ------------

    def _crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """
        Simple set-based crossover:
        - Take a random subset of parent1 + random subset of parent2
        - Remove duplicates
        - Enforce limits
        """
        if self.rng.random() > self.config.crossover_rate:
            # no crossover, just clone one parent
            return parent1[:] if self.rng.random() < 0.5 else parent2[:]

        # Random split points
        cut1 = self.rng.randint(1, max(1, len(parent1) - 1)) if len(parent1) > 1 else 1
        cut2 = self.rng.randint(1, max(1, len(parent2) - 1)) if len(parent2) > 1 else 1

        child_raw = parent1[:cut1] + parent2[cut2:]
        # Deduplicate maintaining order
        seen = set()
        child = []
        for kw in child_raw:
            if kw not in seen:
                seen.add(kw)
                child.append(kw)

        # If too long chars-wise, trim from the end
        while child and self._char_count(child) > self.config.max_chars:
            child.pop()

        # If too few keywords, try to pad from pool
        if len(child) < self.config.min_keywords:
            extras = [kw for kw in self.pool if kw not in seen]
            self.rng.shuffle(extras)
            for kw in extras:
                candidate = child + [kw]
                if len(candidate) > self.config.max_keywords:
                    break
                if self._char_count(candidate) > self.config.max_chars:
                    continue
                child = candidate

        return child

    # ------------ Mutation ------------

    def _mutate(self, individual: List[str]) -> List[str]:
        """
        Mutate an individual by:
        - randomly replacing a keyword
        - or swapping two keywords
        """
        if self.rng.random() > self.config.mutation_rate or not individual:
            return individual

        indiv = individual[:]

        # 50% chance replace, 50% chance swap
        if self.rng.random() < 0.5 and len(self.pool) > 0:
            # Replace a random position with a random pool keyword
            idx = self.rng.randrange(len(indiv))
            replacement_choices = [kw for kw in self.pool if kw not in indiv]
            if replacement_choices:
                new_kw = self.rng.choice(replacement_choices)
                candidate = indiv[:]
                candidate[idx] = new_kw
                if self._char_count(candidate) <= self.config.max_chars:
                    indiv = candidate
        else:
            # Swap two positions
            if len(indiv) > 1:
                i, j = self.rng.sample(range(len(indiv)), 2)
                indiv[i], indiv[j] = indiv[j], indiv[i]

        return indiv

    # ------------ Main GA loop ------------

    def run(self) -> List[str]:
        """
        Run the full GA and return the best individual found.
        """
        population = self.make_initial_population()
        if not population:
            return []

        best_individual = None
        best_fitness = float("-inf")

        for gen in range(self.config.generations):
            fitnesses = [self.fitness(ind) for ind in population]

            # Track best
            for ind, fit in zip(population, fitnesses):
                if fit > best_fitness:
                    best_fitness = fit
                    best_individual = ind

            # Print a tiny bit of debug info (optional, comment out if noisy)
            # print(f"Gen {gen}: best fitness={best_fitness:.2f}, indiv size={len(best_individual)}")

            # Create new population
            new_population: List[List[str]] = []

            while len(new_population) < self.config.population_size:
                # Select parents
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)

                # Crossover
                child = self._crossover(parent1, parent2)

                # Mutation
                child = self._mutate(child)

                # Ensure child obeys basic constraints
                if child and self._within_limits(child):
                    new_population.append(child)

            population = new_population

        return best_individual or []
