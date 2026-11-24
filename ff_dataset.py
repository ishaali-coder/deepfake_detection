"""Dataset that reads a frames CSV (frames_small.csv or frames_balanced.csv).

label 0 = real, label 1 = fake. Paths in the CSV are relative to `root`.

    from ff_dataset import FFDataset
    train_set = FFDataset("/content/data", "train", transform=transform)
"""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class FFDataset(Dataset):
    def __init__(self, root, split, csv_name="frames_balanced.csv", transform=None):
        self.root = Path(root)
        self.transform = transform
        with open(self.root / csv_name) as f:
            self.rows = [(r["path"], int(r["label"]))
                         for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split '{split}' in {self.root / csv_name}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        img = Image.open(self.root / path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
