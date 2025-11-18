from youtube_keyword_bot.data_loader import DataLoader
from youtube_keyword_bot.text_processing import TextProcessing
from youtube_keyword_bot.heuristics import Heuristics


def test_data_loader():
    loader = DataLoader("data/ai_dataset.csv")
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
    loader = DataLoader("data/ai_dataset.csv")
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



def main():
    # Enable one test at a time:

    # test_data_loader()
    # test_text_processing()
    test_heuristics()


if __name__ == "__main__":
    main()
