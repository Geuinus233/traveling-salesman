"""English text classification datasets (extended training sets)."""

from __future__ import annotations

import csv
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ssl
from sklearn.datasets import fetch_20newsgroups

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


@dataclass(frozen=True)
class TextDataset:
    key: str
    name: str
    description: str
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    target_names: list[str]

    @property
    def n_train(self) -> int:
        return len(self.train_texts)

    @property
    def n_test(self) -> int:
        return len(self.test_texts)


def _load_20newsgroups(categories: list[str]) -> TextDataset:
    train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    return TextDataset(
        key=";".join(categories),
        name=f"20 Newsgroups ({len(categories)} categories)",
        description="Classic English newsgroup posts; full sklearn train/test split.",
        train_texts=train.data,
        train_labels=train.target.tolist(),
        test_texts=test.data,
        test_labels=test.target.tolist(),
        target_names=list(train.target_names),
    )


def dataset_newsgroups_10() -> TextDataset:
    cats = [
        "sci.space",
        "sci.med",
        "rec.sport.hockey",
        "rec.sport.baseball",
        "talk.politics.guns",
        "talk.politics.misc",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "alt.atheism",
        "soc.religion.christian",
    ]
    return _load_20newsgroups(cats)


NEWGROUPS_20_CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


def dataset_newsgroups_20() -> TextDataset:
    return _load_20newsgroups(NEWGROUPS_20_CATEGORIES)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urllib.request.urlopen(url, timeout=120) as resp:
        dest.write_bytes(resp.read())


def dataset_ag_news() -> TextDataset:
    """AG News: ~120k English news headlines + descriptions (4 classes)."""
    base = "https://raw.githubusercontent.com/mhjabreel/CharCNN_for_Text_Categorization/master/data/ag_news_csv"
    cache = CACHE_DIR / "ag_news"
    train_path = cache / "train.csv"
    test_path = cache / "test.csv"
    _download(f"{base}/train.csv", train_path)
    _download(f"{base}/test.csv", test_path)

    labels_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    target_names = [labels_map[i] for i in range(1, 5)]

    def read_csv(path: Path) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                label = int(row[0]) - 1
                text = row[1] + " " + row[2] if len(row) > 2 else row[1]
                texts.append(text)
                labels.append(label)
        return texts, labels

    train_texts, train_labels = read_csv(train_path)
    test_texts, test_labels = read_csv(test_path)
    return TextDataset(
        key="ag_news",
        name="AG News (4 classes)",
        description="Large English news corpus (~120k train articles).",
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        target_names=target_names,
    )


def dataset_sms_spam() -> TextDataset:
    """SMS Spam Collection: English SMS ham/spam (~5.5k messages)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    cache = CACHE_DIR / "sms"
    zip_path = cache / "smsspamcollection.zip"
    _download(url, zip_path)

    raw = zipfile.ZipFile(zip_path).read("SMSSpamCollection").decode("utf-8", errors="replace")
    texts, labels = [], []
    for line in raw.splitlines():
        if not line.strip():
            continue
        label_char, _, body = line.partition("\t")
        labels.append(0 if label_char.strip().lower() == "ham" else 1)
        texts.append(body.strip())

    rng = random.Random(42)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    return TextDataset(
        key="sms_spam",
        name="SMS Spam Collection",
        description="English SMS messages; ham vs spam (~4.4k train).",
        train_texts=[texts[i] for i in train_idx],
        train_labels=[labels[i] for i in train_idx],
        test_texts=[texts[i] for i in test_idx],
        test_labels=[labels[i] for i in test_idx],
        target_names=["ham", "spam"],
    )


DATASET_REGISTRY: dict[str, Callable[[], TextDataset]] = {
    "newsgroups_10": dataset_newsgroups_10,
    "newsgroups_20": dataset_newsgroups_20,
    "ag_news": dataset_ag_news,
    "sms_spam": dataset_sms_spam,
}

DATASET_LABELS = {
    "newsgroups_10": "20 Newsgroups — 10 categories",
    "newsgroups_20": "20 Newsgroups — all 20 categories",
    "ag_news": "AG News (large, English)",
    "sms_spam": "SMS Spam (English)",
}


def load_dataset(key: str) -> TextDataset:
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {key}")
    return DATASET_REGISTRY[key]()
