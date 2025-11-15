from youtube_keyword_bot.data_loader import DataLoader
from youtube_keyword_bot.text_processing import TextProcessing


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


def main():
    # Enable one test at a time:

    # test_data_loader()
    test_text_processing()


if __name__ == "__main__":
    main()
