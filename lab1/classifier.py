import json
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config import SENTENCES_FILE
from nlp_processor import NLPProcessor, TextCleaner


class TextClassifier:
    def __init__(self, sentences_file: str = SENTENCES_FILE):
        self.sentences_file = sentences_file
        self.model = None
        self.label_encoder = LabelEncoder()
        self.cleaner = TextCleaner()
        self.nlp = NLPProcessor()

    def _load_data(self):
        if not os.path.exists(self.sentences_file):
            raise FileNotFoundError("Brak pliku sentences.json — dodaj najpierw dane przez /task lub /full_pipeline")
        with open(self.sentences_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        if len(records) < 2:
            raise ValueError("Za mało danych treningowych — dodaj więcej zdań przez /task lub /full_pipeline")
        return [r["text"] for r in records], [r["class"] for r in records]

    def _preprocess(self, text: str) -> str:
        cleaned = self.cleaner.clean(text)
        tokens = self.nlp.tokenize(cleaned)
        tokens = [t for t in tokens if t not in self.nlp.stop_words]
        tokens = [self.nlp.lemmatizer.lemmatize(t, pos="v") for t in tokens]
        return " ".join(tokens)

    def train(self, use_preprocessing: bool = False) -> int:
        texts, labels = self._load_data()
        if use_preprocessing:
            texts = [self._preprocess(t) for t in texts]
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.model = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        self.model.fit(texts, encoded_labels)
        return len(texts)

    def predict(self, text: str, use_preprocessing: bool = False) -> str:
        if self.model is None:
            raise RuntimeError("Model nie jest wytrenowany")
        if use_preprocessing:
            text = self._preprocess(text)
        encoded = self.model.predict([text])[0]
        return self.label_encoder.inverse_transform([encoded])[0]

    def predict_proba(self, text: str, use_preprocessing: bool = False) -> dict:
        if self.model is None:
            raise RuntimeError("Model nie jest wytrenowany")
        if use_preprocessing:
            text = self._preprocess(text)
        proba = self.model.predict_proba([text])[0]
        classes = self.label_encoder.inverse_transform(range(len(proba)))
        return dict(zip(classes, [round(float(p), 4) for p in proba]))
