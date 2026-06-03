"""Lexicon-based dictionaries and feature extractor for NLP classification."""

from __future__ import annotations

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_DICTIONARIES: dict[str, list[str]] = {
    "Sentiment Positive": [
        "good", "great", "excellent", "love", "amazing", "best", "wonderful", "nice",
        "happy", "support", "helpful", "solved", "solve", "solution", "agree", "perfect"
    ],
    "Sentiment Negative": [
        "bad", "worst", "hate", "terrible", "awful", "horrible", "poor", "waste",
        "issue", "bug", "error", "fail", "failed", "crash", "slow", "problem", "disagree"
    ],
    "Science & Space": [
        "space", "nasa", "orbit", "launch", "rocket", "mission", "medical", "disease",
        "health", "science", "scientific", "doctor", "treatment", "patient", "clinical",
        "physician", "technology", "research", "universe", "planet", "astronomy"
    ],
    "Sports": [
        "game", "play", "player", "team", "baseball", "hockey", "win", "won", "cup",
        "coach", "season", "league", "match", "score", "referee", "championship", "stadium"
    ],
    "Politics & Guns": [
        "gun", "guns", "weapon", "firearm", "law", "police", "state", "government",
        "president", "rights", "crime", "court", "war", "military", "federal",
        "senate", "congress", "citizen", "illegal", "legal", "control"
    ],
    "Religion & Atheism": [
        "god", "atheist", "christian", "bible", "jesus", "church", "faith", "belief",
        "religion", "priest", "moral", "sin", "truth", "clergyman", "christ", "prophet",
        "divine", "worship", "theism", "islam", "jewish"
    ],
    "Spam & Promo": [
        "free", "offer", "cash", "prize", "win", "promo", "bonus", "mobile", "txt",
        "reply", "urgent", "claim", "alert", "guaranteed", "call", "stop", "claim",
        "rate", "credit", "card", "expire"
    ],
    "Business & Finance": [
        "business", "company", "market", "stock", "economy", "dollar", "profit", "loss",
        "bank", "money", "price", "industry", "shares", "growth", "revenue", "invest",
        "commercial", "finance", "executive", "ceo"
    ]
}


class DictionaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """Computes density features (word count / total words) for a set of lexical dictionaries."""

    def __init__(self, dictionaries: dict[str, list[str]] | None = None):
        self.dictionaries = dictionaries if dictionaries is not None else DEFAULT_DICTIONARIES
        # Use lower-cased sets for quick lookup
        self._dict_sets = {
            name: set(word.lower() for word in word_list)
            for name, word_list in self.dictionaries.items()
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> np.ndarray:
        # X is an iterable of strings (documents)
        features = []
        for text in X:
            if not isinstance(text, str):
                text = str(text)
            # Tokenize into lowercased words
            tokens = re.findall(r"\b\w+\b", text.lower())
            total_words = max(1, len(tokens))

            row = []
            # Maintain a sorted order of keys to ensure consistent feature indices
            for key in sorted(self._dict_sets.keys()):
                word_set = self._dict_sets[key]
                match_count = sum(1 for t in tokens if t in word_set)
                # Compute frequency density
                row.append(match_count / total_words)
            features.append(row)

        return np.array(features)
