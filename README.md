# YouTube Keyword Optimization Bot

*A heuristic-driven + genetic-algorithm system for generating high-quality YouTube tags*

**Project Overview**

The YouTube Keyword Optimization Bot is an AI-assisted tool designed to automatically generate optimized keyword tags for YouTube videos.
Given:

- A video title
- Optional content categories
- A dataset of existing YouTube videos (titles, categories, tags, etc.)
 
The system searches for thematically similar videos, extracts relevant keywords, expands and filters them semantically, and finally uses a Genetic Algorithm (GA) to evolve a polished, YouTube-ready set of 20–30 tags within the 500-character constraint.
The goal is to simulate how a professional creator or SEO specialist might choose tags.

**Key Features**

Dataset-aware Heuristics Engine

Scores similarity using:

- Keyword overlap
- Noun overlap 
- Category matching 
- Light semantic similarity 
- Hard override for identical titles 
- Produces the initial candidate keyword pool.

**Semantic Expansion**

*Uses spaCy word embeddings to expand keywords based on semantic similarity to nouns in the title.*

**Keyword Pool Filtering**

*Removes low-relevance tokens and prioritizes highly relevant, title-aligned keywords.*

**Genetic Algorithm Optimization**

Evolves the tag list using:

- Tournament selection

- Weighted fitness based on tag quality

- Character budget fitness

- Crossover & mutation

Produces a final optimized keyword list.

**Interactive Command-Line Interface (CLI)**

Allows users to input:
- A custom YouTube video title

- Categories (typed or selected by index)

And see the optimized tags instantly.