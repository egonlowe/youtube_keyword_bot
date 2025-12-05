from youtube_keyword_bot.data_loader import DataLoader
from youtube_keyword_bot.text_processing import TextProcessing
from youtube_keyword_bot.heuristics import Heuristics
from youtube_keyword_bot.keyword_bot import KeywordBot



def test_data_loader():
    loader = DataLoader("data/updatedDataSet.csv")
    dataset = loader.load_dataset()

    print(f"Loaded {len(dataset)} entries.")
    print("First entry:")
    print(dataset[0])


def test_text_processing():
    tp = TextProcessing()

    tokens1 = tp.tokenize("sonic 3's subtle storytelling")
    tokens2 = tp.tokenize("50 useless sonic heroes facts")

    v1 = tp.vectorize(tokens1)
    v2 = tp.vectorize(tokens2)

    print("Tokens 1:", tokens1)
    print("Tokens 2:", tokens2)
    print("Similarity:", tp.similarity(v1, v2))

def test_heuristics():
    print("\n--- Running Heuristics Test ---")

    # 1. Load dataset
    loader = DataLoader("data/updatedDataSet.csv")
    dataset = loader.load_dataset()

    # 2. Initialize helpers
    tp = TextProcessing()
    heur = Heuristics(tp)

    # 3. Sample input title + categories (you can change this later)
    input_title = "the music theory of super mario rpg's battle music"
    input_categories = ["gaming|analysis", "music" ]

    # 4. Compute similarity scores
    scored = heur.score_dataset(dataset, input_title, input_categories)


    print("\nTop 5 most similar videos:")
    for entry, score in scored[:5]:
        print(f"{score:.4f}  |  {entry.title}")

    # 5. Build keyword pool
    keyword_pool = heur.build_keyword_pool(scored, top_n=5)

    print("\nKeyword Pool Preview (first 25):")
    print(keyword_pool[:25])
    print(f"\nTotal pooled keywords: {len(keyword_pool)}")

    print("\n--- End Heuristics Test ---\n")


# def test_ga_initial_population():
#     loader = DataLoader("data/ai_dataset.csv")
#     dataset = loader.load_dataset()
#
#     tp = TextProcessing()
#     heur = Heuristics(tp)
#
#     input_title = "the music theory of super mario rpg's battle music"
#     input_categories = ["gaming", "analysis", "music"]
#
#     ranked, pool = heur.get_top_n_and_pool(dataset, input_title, input_categories, n=10)
#
#     print(f"\nKeyword pool size: {len(pool)}")
#
#     ga = KeywordGA(pool)
#     indiv = ga.make_random_individual()
#
#     print("\nSample GA individual:")
#     print(indiv)
#     print("Char count:", len(",".join(indiv)))

# def test_ga_run():
#     loader = DataLoader("data/ai_dataset.csv")
#     dataset = loader.load_dataset()
#
#     tp = TextProcessing()
#     heur = Heuristics(tp)
#
#     input_title = "the music theory of super mario rpg's battle music"
#     input_categories = ["gaming", "analysis", "music"]
#
#     top_n, pool = heur.get_top_n_and_pool(dataset, input_title, input_categories, n=10)
#
#     print("\n--- GA Test ---")
#     print(f"Keyword pool size from heuristics: {len(pool)}")
#
#     ga = KeywordGA(pool)
#     best = ga.run()
#
#     print("\nBest GA individual:")
#     print(best)
#     print("Keyword count:", len(best))
#     print("Char count:", len(",".join(best)))
#     print("--- End GA Test ---")

def test_ga_run():
    loader = DataLoader("data/ai_dataset.csv")
    dataset = loader.load_dataset()

    bot = KeywordBot(dataset)

    input_title = "the music theory of super mario rpg's battle music"
    input_categories = ["gaming", "analysis", "music"]

    print("\n--- GA Test via KeywordBot ---")
    keywords = bot.generate_keywords(input_title, input_categories)

    print("Final keywords:")
    print(keywords)
    print("Keyword count:", len(keywords))
    print("Char count:", len(",".join(keywords)))
    print("--- End GA Test ---")


def run_cli():
    """
    Simple command-line interface for demo:
      - ask user for title + categories
      - generate keywords with KeywordBot
      - print results + character count
    """
    # Load dataset and bot once
    loader = DataLoader("data/updatedDataSet.csv")
    dataset = loader.load_dataset()
    bot = KeywordBot(dataset)

    # Build a sorted list of unique categories from the dataset
    all_cats = sorted({c for entry in dataset for c in entry.categories})

    print("===============================================")
    print("   YouTube Keyword Generation Bot (CLI Demo)   ")
    print("===============================================\n")
    print("This tool suggests YouTube tags based on your")
    print("video title + categories using heuristics and")
    print("a genetic algorithm.\n")
    print("Type 'q' at any prompt to quit.\n")

    # Show available categories once
    print("Available categories from the dataset:")
    for i, cat in enumerate(all_cats, start=1):
        print(f"  {i:2d}. {cat}")
    print()

    while True:
        title = input("\nEnter video title (or 'q' to quit): ").strip()
        if title.lower() == "q":
            print("Exiting. Goodbye!")
            break

        if not title:
            print("Please enter a non-empty title.")
            continue

        print("\nEnter categories (comma-separated).")
        print("You can type names (e.g., gaming, analysis, music)")
        print("or indices (e.g., 1,3,5). Leave blank for none.")
        cat_input = input("Categories: ").strip()
        if cat_input.lower() == "q":
            print("Exiting. Goodbye!")
            break

        # Parse categories: match by name or index
        chosen_cats: list[str] = []
        if cat_input:
            parts = [p.strip() for p in cat_input.split(",") if p.strip()]
            lower_map = {c.lower(): c for c in all_cats}

            for p in parts:
                # If it's a number, treat as index
                if p.isdigit():
                    idx = int(p)
                    if 1 <= idx <= len(all_cats):
                        chosen_cats.append(all_cats[idx - 1])
                    else:
                        print(f"  [!] Ignoring invalid index: {p}")
                else:
                    # Match by name (case-insensitive)
                    key = p.lower()
                    if key in lower_map:
                        chosen_cats.append(lower_map[key])
                    else:
                        print(f"  [!] Unknown category name: {p}")

        print("\nGenerating keywords...")
        keywords = bot.generate_keywords(title, chosen_cats)

        if not keywords:
            print("No keywords could be generated. Try a different title or categories.")
            continue

        keyword_str = ",".join(keywords)
        char_count = len(keyword_str)

        print("\n-----------------------------------------------")
        print("Suggested Keywords:")
        print(keyword_str)
        print("-----------------------------------------------")
        print(f"Keyword count : {len(keywords)}")
        print(f"Character count: {char_count} / 500")
        print("-----------------------------------------------")





def main():
    # Enable one test at a time:

    # test_data_loader()
    # test_text_processing()
    # test_heuristics()
    # test_ga_initial_population()
    # test_ga_run()
    run_cli()



if __name__ == "__main__":
    main()
