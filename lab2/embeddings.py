import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as gensim_api

GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"
EMBEDDING_METHODS = ["bow", "tfidf", "word2vec", "glove"]

_glove_model = None


def _tokenize(texts: list[str]) -> list[list[str]]:
    return [re.sub(r"[^a-z\s]", "", t.lower()).split() for t in texts]


def get_bow(texts: list[str]):
    vec = CountVectorizer(max_features=10000)
    X = vec.fit_transform(texts)
    return X, vec


def get_tfidf(texts: list[str]):
    vec = TfidfVectorizer(max_features=10000)
    X = vec.fit_transform(texts)
    return X, vec


def get_word2vec(texts: list[str]) -> tuple[np.ndarray, Word2Vec]:
    tokenized = _tokenize(texts)
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5,
                     min_count=2, workers=4, epochs=5)
    X = np.array([
        np.mean([model.wv[w] for w in tokens if w in model.wv] or [np.zeros(100)], axis=0)
        for tokens in tokenized
    ])
    return X, model


def get_glove(texts: list[str]) -> tuple[np.ndarray, KeyedVectors]:
    global _glove_model
    if _glove_model is None:
        print("Pobieranie GloVe ...")
        _glove_model = gensim_api.load(GLOVE_MODEL_NAME)

    tokenized = _tokenize(texts)
    X = np.array([
        np.mean([_glove_model[w] for w in tokens if w in _glove_model] or [np.zeros(100)], axis=0)
        for tokens in tokenized
    ])
    return X, _glove_model


def get_embedding(method: str, texts: list[str]):
    if method == "bow":
        return get_bow(texts)
    elif method == "tfidf":
        return get_tfidf(texts)
    elif method == "word2vec":
        return get_word2vec(texts)
    elif method == "glove":
        return get_glove(texts)
    else:
        raise ValueError(f"Nieznana metoda embeddingu: {method}")


def get_similar_words(model, queries: list[str], topn: int = 10) -> dict[str, list]:
    results = {}
    wv = model.wv if hasattr(model, "wv") else model
    for word in queries:
        try:
            similar = wv.most_similar(word, topn=topn)
            results[word] = similar
        except KeyError:
            results[word] = []
    return results
