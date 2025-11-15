from youtube_keyword_bot.data_loader import DataLoader

def main():

    #test to ensure data is loaded and parsed correctly
    loader = DataLoader("../data/ai_dataset.csv")
    dataset = loader.load_dataset()

    print(f"Loaded {len(dataset)} entries.")
    print("First entry:")
    print(dataset[0])

if __name__ == "__main__":
    main()
