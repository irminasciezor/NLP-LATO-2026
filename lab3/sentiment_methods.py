"""
sentiment_methods.py
--------------------
Wszystkie metody klasyfikacji sentymentu dostępne przez /sentiment.

Każda metoda zwraca SentimentResult:
    label       – etykieta (str)
    score       – pewność/prawdopodobieństwo (float | None)
    scores      – pełny rozkład {etykieta: float} (dict | None)
    model_name  – czytelna nazwa modelu (str)
    method_key  – klucz metody, np. "rule", "nb" (str)
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import re
from dataclasses import dataclass
from typing import Optional
from textblob import TextBlob
from transformers import pipeline as hf_pipeline

import pandas as pd


# ── Wynik ────────────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    label:      str
    model_name: str
    method_key: str
    score:      Optional[float]        = None   # główne prawdopodobieństwo / ocena
    scores:     Optional[dict]         = None   # pełny rozkład, jeśli dostępny


# ── 1. Rule-based ─────────────────────────────────────────────────────────────

_POS_WORDS = {
    "dobry", "świetny", "wspaniały", "genialny", "rewelacyjny", "cudowny",
    "fantastyczny", "doskonały", "piękny", "uwielbiam", "kocham", "polecam",
    "zachwycony", "zadowolony", "idealny", "najlepszy", "super", "ekstra",
    "perfekcyjny", "znakomity", "przepiękny", "rewelacja", "zachwyt",
}
_NEG_WORDS = {
    "zły", "fatalny", "okropny", "koszmarny", "beznadziejny", "tragiczny",
    "straszny", "nienawidzę", "rozczarowany", "żenujący", "skandaliczny",
    "kiepski", "słaby", "nie polecam", "najgorszy", "do niczego",
    "katastrofa", "porażka", "zawiódł", "zawiodła", "zawiodło", "zawiedli",
}


def classify_rule(text: str) -> SentimentResult:
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    pos = len(tokens & _POS_WORDS)
    neg = len(tokens & _NEG_WORDS)

    if pos > neg:
        label, score = "pozytywny", pos / max(pos + neg, 1)
    elif neg > pos:
        label, score = "negatywny", neg / max(pos + neg, 1)
    else:
        label, score = "neutralny", None

    return SentimentResult(
        label=label,
        score=score,
        model_name="Rule-based (słownik)",
        method_key="rule",
    )


# ── 2. Naive Bayes ────────────────────────────────────────────────────────────

def classify_nb(text: str, dataset_path: str = "sentiment_dataset.csv") -> SentimentResult:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    df = _load_dataset(dataset_path)
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf",   MultinomialNB()),
    ])
    model.fit(df["text"], y)

    idx   = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = le.inverse_transform(range(len(proba)))

    return SentimentResult(
        label=le.inverse_transform([idx])[0],
        score=round(float(max(proba)), 4),
        scores={cls: round(float(p), 4) for cls, p in zip(classes, proba)},
        model_name="Naive Bayes (TF-IDF)",
        method_key="nb",
    )


# ── 3. Random Forest ──────────────────────────────────────────────────────────

def classify_rf(text: str, dataset_path: str = "sentiment_dataset.csv") -> SentimentResult:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    df = _load_dataset(dataset_path)
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf",   RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ])
    model.fit(df["text"], y)

    idx   = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = le.inverse_transform(range(len(proba)))

    return SentimentResult(
        label=le.inverse_transform([idx])[0],
        score=round(float(max(proba)), 4),
        scores={cls: round(float(p), 4) for cls, p in zip(classes, proba)},
        model_name="Random Forest (TF-IDF)",
        method_key="rf",
    )


# ── 4. Transformer ────────────────────────────────────────────────────────────

_TRANSFORMER_LABEL_MAP = {
    "LABEL_0": "negatywny",
    "LABEL_1": "neutralny",
    "LABEL_2": "pozytywny",
    "NEGATIVE": "negatywny",
    "NEUTRAL":  "neutralny",
    "POSITIVE": "pozytywny",
    "negative": "negatywny",
    "neutral":  "neutralny",
    "positive": "pozytywny",
}
_transformer_pipeline = None   # lazy init


def classify_transformer(text: str) -> SentimentResult:
    global _transformer_pipeline
    if _transformer_pipeline is None:
        _transformer_pipeline = hf_pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
        )

    result = _transformer_pipeline(text)[0]
    raw_label = result["label"]
    label = _TRANSFORMER_LABEL_MAP.get(raw_label, raw_label.lower())
    score = round(float(result["score"]), 4)

    return SentimentResult(
        label=label,
        score=score,
        model_name="Transformer (twitter-roberta-base-sentiment)",
        method_key="transformer",
    )


# ── 5. TextBlob ───────────────────────────────────────────────────────────────

def classify_textblob(text: str) -> SentimentResult:

    polarity = TextBlob(text).sentiment.polarity  # [-1, 1]

    if polarity > 0.1:
        label = "pozytywny"
    elif polarity < -0.1:
        label = "negatywny"
    else:
        label = "neutralny"

    return SentimentResult(
        label=label,
        score=round(float(polarity), 4),
        model_name="TextBlob (polarity)",
        method_key="textblob",
    )


# ── 6. Stanza ─────────────────────────────────────────────────────────────────

_stanza_nlp = None   # lazy init


def classify_stanza(text: str) -> SentimentResult:
    global _stanza_nlp
    if _stanza_nlp is None:
        import stanza
        stanza.download("en", processors="tokenize,sentiment", verbose=False)
        _stanza_nlp = stanza.Pipeline(
            "en", processors="tokenize,sentiment", verbose=False
        )

    doc = _stanza_nlp(text)
    scores_raw = [sent.sentiment for sent in doc.sentences]  # 0=neg, 1=neu, 2=pos

    avg = sum(scores_raw) / len(scores_raw) if scores_raw else 1
    if avg >= 1.5:
        label = "pozytywny"
    elif avg <= 0.5:
        label = "negatywny"
    else:
        label = "neutralny"

    return SentimentResult(
        label=label,
        score=round(float(avg), 4),
        model_name="Stanza (sentiment)",
        method_key="stanza",
    )


# ── 7–9. Modele sekwencyjne (SimpleRNN / LSTM / GRU) ─────────────────────────

def _classify_sequential(text: str, method_key: str) -> SentimentResult:
    from sentiment_classifier import (
        GRUSentimentClassifier,
        LSTMSentimentClassifier,
        SimpleRNNSentimentClassifier,
    )

    clf_map = {
        "simplernn": SimpleRNNSentimentClassifier,
        "lstm":      LSTMSentimentClassifier,
        "gru":       GRUSentimentClassifier,
    }
    name_map = {
        "simplernn": "SimpleRNN",
        "lstm":      "LSTM",
        "gru":       "GRU",
    }

    clf = clf_map[method_key]()

    if not clf.is_trained():
        raise RuntimeError(
            f"Model {name_map[method_key]} nie jest wytrenowany.\n"
            f"Uruchom najpierw `/train_{method_key}`."
        )

    label  = clf.predict(text)
    scores = clf.predict_proba(text)
    score  = scores.get(label)

    return SentimentResult(
        label=label,
        score=score,
        scores=scores,
        model_name=f"Sieć neuronowa ({name_map[method_key]})",
        method_key=method_key,
    )


def classify_simplernn(text: str) -> SentimentResult:
    return _classify_sequential(text, "simplernn")


def classify_lstm(text: str) -> SentimentResult:
    return _classify_sequential(text, "lstm")


def classify_gru(text: str) -> SentimentResult:
    return _classify_sequential(text, "gru")


# ── Dispatcher ────────────────────────────────────────────────────────────────

METHODS: dict[str, callable] = {
    "rule":        classify_rule,
    "nb":          classify_nb,
    "rf":          classify_rf,
    "transformer": classify_transformer,
    "textblob":    classify_textblob,
    "stanza":      classify_stanza,
    "simplernn":   classify_simplernn,
    "lstm":        classify_lstm,
    "gru":         classify_gru,
}


def classify(method: str, text: str) -> SentimentResult:
    """
    Główny punkt wejścia.

    Args:
        method: klucz metody (rule | nb | rf | transformer |
                              textblob | stanza | simplernn | lstm | gru)
        text:   tekst do klasyfikacji

    Returns:
        SentimentResult

    Raises:
        ValueError: nieznana metoda
        RuntimeError: model niezaładowany / brak treningu
    """
    method = method.lower().strip()
    if method not in METHODS:
        available = " | ".join(METHODS.keys())
        raise ValueError(f"Nieznana metoda: `{method}`\nDostępne: {available}")
    return METHODS[method](text)


# ── Helper ────────────────────────────────────────────────────────────────────

def _load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Nie znaleziono datasetu: `{path}`\n"
            "Upewnij się, że plik `sentiment_dataset.csv` istnieje w katalogu bota."
        )
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset musi zawierać kolumny: text, label")
    return df.dropna(subset=["text", "label"])
