import csv
from pathlib import Path

class VideoEntry:
    def __init__(self, title, categories, keywords):
        self.title = title
        self.categories = categories
        self.keywords = keywords

    def __repr__(self):
        return f"VideoEntry(title={self.title}, categories={self.categories}, keywords={len(self.keywords)} kws)"


class DataLoader:
    def __init__(self, csv_path: str):
        # Step 1: go from data_loader.py → youtube_keyword_bot/ → src/ → project root
        project_root = Path(__file__).resolve().parent.parent.parent

        # Step 2: build full path to CSV inside /data/
        self.csv_path = (project_root / csv_path).resolve()

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

