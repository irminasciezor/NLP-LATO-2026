from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


class Visualizer:
    @staticmethod
    def _filename(prefix: str) -> str:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        return f"{prefix}_{ts}.png"

    def plot_histogram(self, text: str) -> str:
        tokens = word_tokenize(text)
        lengths = [len(t) for t in tokens if t.isalpha()]
        path = self._filename("Sentence")
        plt.figure(figsize=(8, 4))
        plt.hist(lengths, bins=range(1, max(lengths) + 2), color="steelblue", edgecolor="white", align="left")
        plt.title("Histogram długości tokenów")
        plt.xlabel("Długość tokenu")
        plt.ylabel("Liczba wystąpień")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def plot_wordcloud(self, text: str) -> str:
        path = self._filename("Sentence")
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def plot_bar(self, text: str) -> str:
        tokens = word_tokenize(text)
        words = [t.lower() for t in tokens if t.isalpha()]
        freq = Counter(words).most_common(10)
        labels, values = zip(*freq) if freq else ([], [])
        path = self._filename("Sentence")
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color="coral", edgecolor="white")
        plt.title("Najczęstsze tokeny")
        plt.xlabel("Token")
        plt.ylabel("Liczba wystąpień")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def plot_class_counts(self, class_counts: dict) -> str:
        path = self._filename("Sentence")
        labels = list(class_counts.keys())
        values = list(class_counts.values())
        colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple", "gold"]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values, color=colors[:len(labels)], edgecolor="white")
        plt.title("Liczność klas")
        plt.xlabel("Klasa")
        plt.ylabel("Liczba zdań")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def full_pipeline_plots(self, text: str) -> list[str]:
        return [
            self.plot_bar(text),
            self.plot_histogram(text),
            self.plot_wordcloud(text),
        ]

    def stats_plots(self, all_text: str, class_counts: dict) -> list[str]:
        return [
            self.plot_bar(all_text),
            self.plot_histogram(all_text),
            self.plot_wordcloud(all_text),
            self.plot_class_counts(class_counts),
        ]
