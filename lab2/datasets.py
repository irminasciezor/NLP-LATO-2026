import os
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

SUPPORTED_DATASETS = ["20news_group", "imdb", "amazon", "ag_news"]


def load_dataset(name: str) -> tuple[list[str], list, list[str]]:
    if name == "20news_group":
        return _load_20news()
    elif name == "imdb":
        return _load_csv_dataset("imdb.csv", text_col="review", label_col="sentiment")
    elif name == "amazon":
        return _load_csv_dataset("amazon.csv", text_col="reviewText", label_col="overall")
    elif name == "ag_news":
        return _load_csv_dataset("ag_news.csv", text_col="description", label_col="class")
    else:
        raise ValueError(f"Nieznany dataset: {name}. Dostępne: {SUPPORTED_DATASETS}")

def get_query_words(texts: list[str], n: int = 5) -> list[str]:
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))
    words = []
    for text in texts:
        words.extend(re.sub(r"[^a-z\s]", "", text.lower()).split())
    freq = Counter(w for w in words if w not in stop and len(w) > 3)
    return [w for w, _ in freq.most_common(n)]


def _load_20news() -> tuple[list[str], list, list[str]]:
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    return list(data.data), list(data.target), list(data.target_names)


def _load_csv_dataset(filename: str, text_col: str, label_col: str) -> tuple[list[str], list, list[str]]:
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Brak pliku {filename}. Pobierz dataset z Kaggle i umieść w katalogu projektu jako {filename}."
        )
    df = pd.read_csv(path).dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].tolist()
    class_names = [str(c) for c in sorted(set(labels))]
    return texts, labels, class_names
