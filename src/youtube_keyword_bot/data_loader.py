import csv

class VideoEntry:
    def __init__(self, title, categories, keywords):
        self.title = title
        self.categories = categories      # list of categories
        self.keywords = keywords          # list of keywords

    def __repr__(self):
        return f"VideoEntry(title={self.title}, categories={self.categories}, keywords={self.keywords[:5]}...)"

class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_dataset(self):
        entries = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                title = row["title"].strip().lower()

                # Preserve hierarchy in categories
                raw_cats = row["categories"].strip().lower()
                categories = [c.strip() for c in raw_cats.split(";")]

                # Parse keywords
                raw_keywords = row["keywords"].strip().lower()
                keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]

                entry = VideoEntry(title, categories, keywords)
                entries.append(entry)

        return entries

