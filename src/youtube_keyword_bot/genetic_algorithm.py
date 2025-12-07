import random
from dataclasses import dataclass
from typing import List, Dict, Optional


# Stores configuration values used to control population size, constraints, and probabilities in the GA.
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


# Implements a full genetic algorithm for optimizing keyword lists.
class KeywordGA:
    # Initializes the GA with a keyword pool, optional weighting, GA configuration, and random seed.
    def __init__(
        self,
        keyword_pool: List[str],
        keyword_weights: Optional[Dict[str, float]] = None,
        config: Optional[GAConfig] = None,
        rng_seed: Optional[int] = None
    ):
        self.pool = list(dict.fromkeys(keyword_pool))
        self.weights = keyword_weights or {kw: 1.0 for kw in self.pool}
        self.config = config or GAConfig()
        self.rng = random.Random(rng_seed)

    # Converts keyword list into a comma-separated string.
    def _keywords_to_string(self, keywords: List[str]) -> str:
        return ",".join(keywords)

    # Returns total characters used by the keyword list.
    def _char_count(self, keywords: List[str]) -> int:
        return len(self._keywords_to_string(keywords))

    # Checks whether a keyword list meets the GA's min/max and character-limit constraints.
    def _within_limits(self, keywords: List[str]) -> bool:
        if not (self.config.min_keywords <= len(keywords) <= self.config.max_keywords):
            return False
        if self._char_count(keywords) > self.config.max_chars:
            return False
        return True

    # Creates a random keyword list that satisfies basic keyword-count and character constraints.
    def make_random_individual(self) -> List[str]:
        if not self.pool:
            return []

        target_size = self.rng.randint(self.config.min_keywords, self.config.max_keywords)
        shuffled = self.pool[:]
        self.rng.shuffle(shuffled)

        individual: List[str] = []
        for kw in shuffled:
            candidate = individual + [kw]
            if len(candidate) > target_size:
                break
            if self._char_count(candidate) > self.config.max_chars:
                continue
            individual = candidate

        return individual

    # Builds the first population by generating random valid individuals.
    def make_initial_population(self) -> List[List[str]]:
        population = []
        attempts = 0
        while len(population) < self.config.population_size and attempts < self.config.population_size * 5:
            indiv = self.make_random_individual()
            if indiv and self._within_limits(indiv):
                population.append(indiv)
            attempts += 1
            if len(population) == 0 and attempts > self.config.population_size * 2:
                self.config.min_keywords = max(5, self.config.min_keywords - 1)
        return population

    # Computes the fitness value for a single keyword list.
    def fitness(self, individual: List[str]) -> float:
        if not individual:
            return 0.0

        total_chars = self._char_count(individual)
        if total_chars > self.config.max_chars:
            return 0.0

        weight_score = sum(self.weights.get(kw, 1.0) for kw in individual)

        junk_penalty = 0.0
        for kw in individual:
            if len(kw.split()) > 4:
                junk_penalty += 0.5

        weight_score = max(0.0, weight_score - junk_penalty)

        budget_usage = total_chars / self.config.max_chars
        size_score = len(individual) / self.config.max_keywords

        base = weight_score
        bonus = 0.3 * budget_usage + 0.2 * size_score

        return base * (1.0 + bonus)

    # Selects a parent using tournament selection.
    def _tournament_select(self, population: List[List[str]], fitnesses: List[float]) -> List[str]:
        indices = [self.rng.randrange(len(population)) for _ in range(self.config.tournament_size)]
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx][:]

    # Performs crossover between two parent keyword lists to create a new child.
    def _crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        if self.rng.random() > self.config.crossover_rate:
            return parent1[:] if self.rng.random() < 0.5 else parent2[:]

        cut1 = self.rng.randint(1, max(1, len(parent1) - 1)) if len(parent1) > 1 else 1
        cut2 = self.rng.randint(1, max(1, len(parent2) - 1)) if len(parent2) > 1 else 1

        child_raw = parent1[:cut1] + parent2[cut2:]

        seen = set()
        child = []
        for kw in child_raw:
            if kw not in seen:
                seen.add(kw)
                child.append(kw)

        while child and self._char_count(child) > self.config.max_chars:
            child.pop()

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

    # Mutates a keyword list by replacing or swapping keywords.
    def _mutate(self, individual: List[str]) -> List[str]:
        if self.rng.random() > self.config.mutation_rate or not individual:
            return individual

        indiv = individual[:]

        if self.rng.random() < 0.5 and len(self.pool) > 0:
            idx = self.rng.randrange(len(indiv))
            replacement_choices = [kw for kw in self.pool if kw not in indiv]
            if replacement_choices:
                new_kw = self.rng.choice(replacement_choices)
                candidate = indiv[:]
                candidate[idx] = new_kw
                if self._char_count(candidate) <= self.config.max_chars:
                    indiv = candidate
        else:
            if len(indiv) > 1:
                i, j = self.rng.sample(range(len(indiv)), 2)
                indiv[i], indiv[j] = indiv[j], indiv[i]

        return indiv

    # Runs the GA through all generations and returns the highest-fitness keyword list discovered.
    def run(self) -> List[str]:
        population = self.make_initial_population()
        if not population:
            return []

        best_individual = None
        best_fitness = float("-inf")

        for gen in range(self.config.generations):
            fitnesses = [self.fitness(ind) for ind in population]

            for ind, fit in zip(population, fitnesses):
                if fit > best_fitness:
                    best_fitness = fit
                    best_individual = ind

            new_population: List[List[str]] = []

            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)

                child = self._crossover(parent1, parent2)
                child = self._mutate(child)

                if child and self._within_limits(child):
                    new_population.append(child)

            population = new_population

        return best_individual or []
