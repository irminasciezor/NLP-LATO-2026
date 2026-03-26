import json
import os
from collections import Counter

import nltk
from nltk import trigrams

from config import SENTENCES_FILE
from nlp_processor import NLPProcessor


class SentenceStore:
    def __init__(self, filepath: str = SENTENCES_FILE):
        self.filepath = filepath

    def save(self, text: str, label: str) -> None:
        records = []
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    records = json.load(f)
                except json.JSONDecodeError:
                    records = []
        records.append({"text": text, "class": label})
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


class StatsAnalyzer:
    def __init__(self, sentences_file: str = SENTENCES_FILE):
        self.sentences_file = sentences_file
        self.nlp = NLPProcessor()

    def _load_data(self):
        if not os.path.exists(self.sentences_file):
            raise FileNotFoundError("Brak pliku sentences.json")
        with open(self.sentences_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not records:
            raise ValueError("Plik sentences.json jest pusty")
        return records

    def analyze(self) -> dict:
        records = self._load_data()
        all_text = " ".join(r["text"] for r in records)
        tokens = self.nlp.tokenize(all_text)
        words = [t.lower() for t in tokens if t.isalpha()]

        unique_tokens = sorted(set(words))

        bigrams = list(nltk.bigrams(words))
        unique_bigrams = sorted(set(bigrams))
        bigram_freq = Counter(bigrams).most_common(10)

        tri = list(trigrams(words))
        unique_trigrams = sorted(set(tri))
        trigram_freq = Counter(tri).most_common(10)

        class_counts = Counter(r["class"] for r in records)
        word_freq = Counter(words).most_common(10)

        return {
            "num_sentences": len(records),
            "num_tokens": len(tokens),
            "num_unique_tokens": len(unique_tokens),
            "unique_tokens": unique_tokens,
            "num_unique_bigrams": len(unique_bigrams),
            "top_bigrams": bigram_freq,
            "num_unique_trigrams": len(unique_trigrams),
            "top_trigrams": trigram_freq,
            "class_counts": dict(class_counts),
            "word_freq": word_freq,
            "all_text": all_text,
        }
