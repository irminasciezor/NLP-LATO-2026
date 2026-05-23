"""
Klasyfikatory sentymentu oparte o sieci sekwencyjne.

Dostępne klasy:
    SimpleRNNSentimentClassifier  – Embedding → SimpleRNN → Dense → Dense
    LSTMSentimentClassifier       – Embedding → LSTM      → Dense → Dense
    GRUSentimentClassifier        – Embedding → GRU       → Dense → Dense

Każda klasa ma identyczne API:
    clf.train()           # trenuje i zapisuje model do .h5 + tokenizer do .pkl
    clf.predict(text)     # wczytuje model z .h5, zwraca etykietę
    clf.predict_proba(text)  # zwraca {etykieta: prawdopodobieństwo}
    clf.is_trained()      # True jeśli pliki .h5 i .pkl istnieją na dysku
    clf.model_summary()   # tekstowe podsumowanie architektury
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET = os.path.join(_HERE, "sentiment_dataset.csv")

VOCAB_SIZE  = 5_000
MAX_LEN     = 30
EMBED_DIM   = 64
RNN_UNITS   = 64
DENSE_UNITS = 64
EPOCHS      = 15
BATCH_SIZE  = 16
EARLY_STOPPING_PATIENCE = max(1, int(EPOCHS * 0.1))

MODELS_DIR = os.path.join(_HERE, "models")

_DEFAULTS = {
    "simplernn": (
        os.path.join(MODELS_DIR, "simple_rnn_sentiment.h5"),
        os.path.join(MODELS_DIR, "simple_rnn_tokenizer.pkl"),
    ),
    "lstm": (
        os.path.join(MODELS_DIR, "lstm_sentiment.h5"),
        os.path.join(MODELS_DIR, "lstm_tokenizer.pkl"),
    ),
    "gru": (
        os.path.join(MODELS_DIR, "gru_sentiment.h5"),
        os.path.join(MODELS_DIR, "gru_tokenizer.pkl"),
    ),
}


class _BaseSequentialClassifier:
    _model_key: str = ""

    def __init__(
        self,
        dataset_path:            str = DEFAULT_DATASET,
        model_path:              Optional[str] = None,
        tokenizer_path:          Optional[str] = None,
        vocab_size:              int = VOCAB_SIZE,
        max_len:                 int = MAX_LEN,
        embed_dim:               int = EMBED_DIM,
        rnn_units:               int = RNN_UNITS,
        dense_units:             int = DENSE_UNITS,
        epochs:                  int = EPOCHS,
        batch_size:              int = BATCH_SIZE,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    ):
        default_h5, default_pkl = _DEFAULTS.get(self._model_key, ("", ""))
        self.dataset_path   = dataset_path
        self.model_path     = model_path     or default_h5 or os.path.join(MODELS_DIR, f"{self._model_key}_sentiment.h5")
        self.tokenizer_path = tokenizer_path or default_pkl or os.path.join(MODELS_DIR, f"{self._model_key}_tokenizer.pkl")
        self.vocab_size     = vocab_size
        self.max_len        = max_len
        self.embed_dim      = embed_dim
        self.rnn_units      = rnn_units
        self.dense_units    = dense_units
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.early_stopping_patience = early_stopping_patience

        self.model:       Optional[Sequential] = None
        self.tokenizer:   Optional[Tokenizer]  = None
        self.label_index: dict[int, str]       = {}
        self.n_classes:   int                  = 0

    def _load_dataset(self) -> tuple[list[str], list[str]]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Nie znaleziono datasetu: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(f"Dataset musi zawierac kolumny: text, label")
        df = df.dropna(subset=["text", "label"])
        if len(df) < 10:
            raise ValueError(f"Za malo danych: {len(df)} wierszy, wymagane min. 10.")
        return df["text"].tolist(), df["label"].tolist()

    def _texts_to_sequences(self, texts: list[str]) -> np.ndarray:
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")

    def _encode_labels(self, labels: list[str]) -> tuple[np.ndarray, dict[int, str]]:
        unique = sorted(set(labels))
        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique)}
        idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}
        encoded = np.array([label_to_idx[l] for l in labels], dtype=np.int32)
        return encoded, idx_to_label

    def _build_model(self) -> Sequential:
        raise NotImplementedError

    def _save_tokenizer(self):
        with open(self.tokenizer_path, "wb") as f:
            pickle.dump((self.tokenizer, self.label_index, self.n_classes), f)

    def _load_tokenizer(self):
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Nie znaleziono tokenizera: {self.tokenizer_path}")
        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer, self.label_index, self.n_classes = pickle.load(f)

    def _load_model_from_file(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Nie znaleziono pliku modelu: {self.model_path}")
        self.model = load_model(self.model_path)

    def is_trained(self) -> bool:
        return os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path)

    def train(self, verbose: bool = False) -> dict:
        texts, labels = self._load_dataset()

        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)

        encoded_labels, self.label_index = self._encode_labels(labels)
        self.n_classes = len(self.label_index)

        X = self._texts_to_sequences(texts)
        y = encoded_labels

        self.model = self._build_model()
        if verbose:
            self.model.summary()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True,
        )

        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1 if verbose else 0,
        )

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        self._save_tokenizer()

        return {
            "n_samples":  len(texts),
            "n_classes":  self.n_classes,
            "classes":    list(self.label_index.values()),
            "epochs":     len(history.history["loss"]),
            "final_loss": round(float(history.history["loss"][-1]), 4),
            "final_acc":  round(float(history.history["accuracy"][-1]) * 100, 2),
            "model_path": self.model_path,
        }

    def _ensure_loaded(self):
        if self.model is None or self.tokenizer is None:
            self._load_tokenizer()
            self._load_model_from_file()

    def predict(self, text: str) -> str:
        self._ensure_loaded()
        X = self._texts_to_sequences([text])
        idx = int(np.argmax(self.model.predict(X, verbose=0)[0]))
        return self.label_index[idx]

    def predict_proba(self, text: str) -> dict[str, float]:
        self._ensure_loaded()
        X = self._texts_to_sequences([text])
        proba = self.model.predict(X, verbose=0)[0]
        return {
            self.label_index[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

    def model_summary(self) -> str:
        self._ensure_loaded()
        lines: list[str] = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)


class SimpleRNNSentimentClassifier(_BaseSequentialClassifier):
    _model_key = "simplernn"

    def _build_model(self) -> Sequential:
        model = Sequential(name="SimpleRNN_Sentiment", layers=[
            Embedding(input_dim=self.vocab_size + 1, output_dim=self.embed_dim, name="embedding"),
            SimpleRNN(units=self.rnn_units, name="simple_rnn"),
            Dense(units=self.dense_units, activation="relu", name="dense_hidden"),
            Dense(units=self.n_classes, activation="softmax", name="dense_output"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model


class LSTMSentimentClassifier(_BaseSequentialClassifier):
    _model_key = "lstm"

    def _build_model(self) -> Sequential:
        model = Sequential(name="LSTM_Sentiment", layers=[
            Embedding(input_dim=self.vocab_size + 1, output_dim=self.embed_dim, name="embedding"),
            LSTM(units=self.rnn_units, name="lstm"),
            Dense(units=self.dense_units, activation="relu", name="dense_hidden"),
            Dense(units=self.n_classes, activation="softmax", name="dense_output"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model


class GRUSentimentClassifier(_BaseSequentialClassifier):
    _model_key = "gru"

    def _build_model(self) -> Sequential:
        model = Sequential(name="GRU_Sentiment", layers=[
            Embedding(input_dim=self.vocab_size + 1, output_dim=self.embed_dim, name="embedding"),
            GRU(units=self.rnn_units, name="gru"),
            Dense(units=self.dense_units, activation="relu", name="dense_hidden"),
            Dense(units=self.n_classes, activation="softmax", name="dense_output"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model