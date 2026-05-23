"""
Składnia:
    /train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>

Co robi:
    1. Trenuje wskazany model sekwencyjny na sentiment_dataset.csv
    2. Zapisuje model do models/<model>_sentiment.h5
    3. Zapisuje tokenizer i encoder etykiet do models/<model>_tokenizer.pkl
    4. Zwraca podsumowanie treningu
    5. Zwraca ścieżki do zapisanych plików
    6. Generuje dwa wykresy: accuracy i loss (osobne pliki)
"""

from __future__ import annotations

import os
import re
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sentiment_classifier import (
    GRUSentimentClassifier,
    LSTMSentimentClassifier,
    SimpleRNNSentimentClassifier,
)

_HERE       = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR   = os.path.join(_HERE, "lab3plots")
DATASET_CSV = os.path.join(_HERE, "sentiment_dataset.csv")

VALID_MODELS   = ("simplernn", "lstm", "gru")
VALID_DATASETS = ("amazon", "imdb", "custom")

_CLF_MAP = {
    "simplernn": SimpleRNNSentimentClassifier,
    "lstm":      LSTMSentimentClassifier,
    "gru":       GRUSentimentClassifier,
}

_LABEL_MAP = {
    "simplernn": "SimpleRNN",
    "lstm":      "LSTM",
    "gru":       "GRU",
}

_CMD_RE = re.compile(r'^/train\s+model=(\S+)\s+dataset=(\S+)', re.IGNORECASE)


class TrainHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["train"])
        def handle_train(message):
            self._handle(message)

    def _handle(self, message):
        m = _CMD_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage())
            return

        model_key   = m.group(1).lower().strip()
        dataset_key = m.group(2).lower().strip()

        if model_key not in VALID_MODELS:
            self.bot.reply_to(message, f"Nieznany model: {model_key}\nDostepne: {' | '.join(VALID_MODELS)}")
            return

        if dataset_key not in VALID_DATASETS:
            self.bot.reply_to(message, f"Nieznany dataset: {dataset_key}\nDostepne: {' | '.join(VALID_DATASETS)}")
            return

        model_label = _LABEL_MAP[model_key]
        self.bot.reply_to(message, f"Trenuje model {model_label}...")

        try:
            clf, metrics, history = self._train(model_key)
            plot_acc, plot_loss   = self._save_plots(history, model_key, dataset_key, model_label)
            reply = self._format_summary(metrics, clf, model_label, dataset_key, plot_acc, plot_loss)
            self.bot.reply_to(message, reply)

            for plot_path in (plot_acc, plot_loss):
                with open(plot_path, "rb") as img:
                    self.bot.send_photo(message.chat.id, img)

        except FileNotFoundError as e:
            self.bot.reply_to(message, f"Brak pliku: {e}")
        except ValueError as e:
            self.bot.reply_to(message, f"Blad danych: {e}")
        except Exception as e:
            self.bot.reply_to(message, f"Blad: {e}")

    def _train(self, model_key: str):
        clf = _CLF_MAP[model_key](dataset_path=DATASET_CSV)
        texts, labels = clf._load_dataset()

        clf.tokenizer = Tokenizer(num_words=clf.vocab_size, oov_token="<OOV>")
        clf.tokenizer.fit_on_texts(texts)

        encoded_labels, clf.label_index = clf._encode_labels(labels)
        clf.n_classes = len(clf.label_index)

        seqs = clf.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=clf.max_len, padding="post", truncating="post")
        y = encoded_labels

        clf.model = clf._build_model()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=clf.early_stopping_patience,
            restore_best_weights=True,
        )

        history = clf.model.fit(
            X, y,
            epochs=clf.epochs,
            batch_size=clf.batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )

        model_dir = os.path.dirname(clf.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        clf.model.save(clf.model_path)
        with open(clf.tokenizer_path, "wb") as f:
            pickle.dump((clf.tokenizer, clf.label_index, clf.n_classes), f)

        epochs_run = len(history.history["loss"])
        metrics = {
            "n_samples":  len(texts),
            "n_classes":  clf.n_classes,
            "classes":    list(clf.label_index.values()),
            "epochs_run": epochs_run,
            "epochs_max": clf.epochs,
            "final_loss": round(float(history.history["loss"][-1]), 4),
            "final_acc":  round(float(history.history["accuracy"][-1]) * 100, 2),
            "val_loss":   round(float(history.history["val_loss"][-1]), 4),
            "val_acc":    round(float(history.history["val_accuracy"][-1]) * 100, 2),
        }

        return clf, metrics, history.history

    def _save_plots(self, history, model_key, dataset_key, model_label):
        os.makedirs(PLOTS_DIR, exist_ok=True)

        base      = f"train_history_{model_key}_{dataset_key}"
        acc_path  = os.path.join(PLOTS_DIR, f"{base}_accuracy.png")
        loss_path = os.path.join(PLOTS_DIR, f"{base}_loss.png")

        epochs = range(1, len(history["loss"]) + 1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history["accuracy"],     "o-", label="Train accuracy", color="#2196F3")
        ax.plot(epochs, history["val_accuracy"], "s--", label="Val accuracy",  color="#FF9800")
        ax.set_title(f"{model_label} - Accuracy ({dataset_key})")
        ax.set_xlabel("Epoka")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(acc_path, dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history["loss"],     "o-", label="Train loss", color="#4CAF50")
        ax.plot(epochs, history["val_loss"], "s--", label="Val loss",  color="#F44336")
        ax.set_title(f"{model_label} - Loss ({dataset_key})")
        ax.set_xlabel("Epoka")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(loss_path, dpi=120)
        plt.close(fig)

        return acc_path, loss_path

    def _format_summary(self, metrics, clf, model_label, dataset_key, plot_acc, plot_loss):
        classes_str   = ", ".join(metrics["classes"])
        stopped_early = metrics["epochs_run"] < metrics["epochs_max"]
        early_note    = f" (early stopping po {metrics['epochs_run']} epokach)" if stopped_early else ""

        return (
            f"Model {model_label} wytrenowany.\n\n"
            f"Dataset: {dataset_key} -> sentiment_dataset.csv\n"
            f"Klasy: {classes_str}\n"
            f"Probek: {metrics['n_samples']}\n\n"
            f"Trening:\n"
            f"  Epoki: {metrics['epochs_run']}/{metrics['epochs_max']}{early_note}\n"
            f"  Train loss: {metrics['final_loss']} | acc: {metrics['final_acc']}%\n"
            f"  Val loss:   {metrics['val_loss']} | acc: {metrics['val_acc']}%\n\n"
            f"Zapisano:\n"
            f"  Model: {os.path.relpath(clf.model_path, _HERE)}\n"
            f"  Tokenizer: {os.path.relpath(clf.tokenizer_path, _HERE)}\n\n"
            f"Wykresy:\n"
            f"  {os.path.relpath(plot_acc, _HERE)}\n"
            f"  {os.path.relpath(plot_loss, _HERE)}"
        )

    def _usage(self) -> str:
        return (
            "Uzycie:\n"
            "/train model=<model> dataset=<dataset>\n\n"
            f"Modele: {' | '.join(VALID_MODELS)}\n"
            f"Datasety: {' | '.join(VALID_DATASETS)}\n\n"
            "Przyklady:\n"
            "/train model=simplernn dataset=custom\n"
            "/train model=lstm dataset=imdb\n"
            "/train model=gru dataset=amazon"
        )

    def help_section(self) -> str:
        return (
            "---\n"
            "Trening modeli sekwencyjnych\n\n"
            "/train model=<model> dataset=<dataset>\n\n"
            f"Modele: {' | '.join(VALID_MODELS)}\n"
            f"Datasety: {' | '.join(VALID_DATASETS)}\n"
        )