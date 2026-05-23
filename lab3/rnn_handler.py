"""
Komendy:
    /train_rnn       – trenuje SimpleRNN na sentiment_dataset.csv
    /train_lstm      – trenuje LSTM na sentiment_dataset.csv
    /train_gru       – trenuje GRU na sentiment_dataset.csv

    /sentiment_rnn   – klasyfikuje tekst modelem SimpleRNN
    /sentiment_lstm  – klasyfikuje tekst modelem LSTM
    /sentiment_gru   – klasyfikuje tekst modelem GRU

    /rnn_info        – architektura modelu SimpleRNN
    /lstm_info       – architektura modelu LSTM
    /gru_info        – architektura modelu GRU
"""

from __future__ import annotations

import re

from sentiment_classifier import (
    GRUSentimentClassifier,
    LSTMSentimentClassifier,
    SimpleRNNSentimentClassifier,
)

_SENTIMENT_RE = re.compile(r'^/sentiment_\w+\s+(.+)', re.IGNORECASE | re.DOTALL)


class RNNHandler:
    def __init__(self, bot):
        self.bot = bot

        self.classifiers = {
            "rnn":  SimpleRNNSentimentClassifier(),
            "lstm": LSTMSentimentClassifier(),
            "gru":  GRUSentimentClassifier(),
        }

        self._labels = {
            "rnn":  "SimpleRNN",
            "lstm": "LSTM",
            "gru":  "GRU",
        }

        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["train_rnn"])
        def h_train_rnn(msg): self._handle_train(msg, "rnn")

        @self.bot.message_handler(commands=["train_lstm"])
        def h_train_lstm(msg): self._handle_train(msg, "lstm")

        @self.bot.message_handler(commands=["train_gru"])
        def h_train_gru(msg): self._handle_train(msg, "gru")

        @self.bot.message_handler(commands=["sentiment_rnn"])
        def h_sent_rnn(msg): self._handle_sentiment(msg, "rnn")

        @self.bot.message_handler(commands=["sentiment_lstm"])
        def h_sent_lstm(msg): self._handle_sentiment(msg, "lstm")

        @self.bot.message_handler(commands=["sentiment_gru"])
        def h_sent_gru(msg): self._handle_sentiment(msg, "gru")

        @self.bot.message_handler(commands=["rnn_info"])
        def h_info_rnn(msg): self._handle_info(msg, "rnn")

        @self.bot.message_handler(commands=["lstm_info"])
        def h_info_lstm(msg): self._handle_info(msg, "lstm")

        @self.bot.message_handler(commands=["gru_info"])
        def h_info_gru(msg): self._handle_info(msg, "gru")

    def _handle_train(self, message, key: str):
        clf   = self.classifiers[key]
        label = self._labels[key]

        self.bot.reply_to(message, f"Rozpoczynam trening modelu {label}...")

        try:
            metrics = clf.train(verbose=False)
            classes_str = ", ".join(metrics["classes"])

            arch_map = {
                "rnn":  "Embedding -> SimpleRNN -> Dense(relu) -> Dense(softmax)",
                "lstm": "Embedding -> LSTM      -> Dense(relu) -> Dense(softmax)",
                "gru":  "Embedding -> GRU       -> Dense(relu) -> Dense(softmax)",
            }

            reply = (
                f"Model {label} wytrenowany.\n\n"
                f"Architektura: {arch_map[key]}\n\n"
                f"Probek: {metrics['n_samples']}\n"
                f"Klas: {metrics['n_classes']} ({classes_str})\n"
                f"Epok: {metrics['epochs']}\n\n"
                f"Loss: {metrics['final_loss']}\n"
                f"Accuracy: {metrics['final_acc']}%\n\n"
                f"Model: {metrics['model_path']}\n"
                f"Tokenizer: {clf.tokenizer_path}"
            )
            self.bot.reply_to(message, reply)

        except FileNotFoundError as e:
            self.bot.reply_to(message, f"Blad - brak pliku: {e}")
        except ValueError as e:
            self.bot.reply_to(message, f"Blad danych: {e}")
        except Exception as e:
            self.bot.reply_to(message, f"Blad treningu: {e}")

    def _handle_sentiment(self, message, key: str):
        clf   = self.classifiers[key]
        label = self._labels[key]

        m = _SENTIMENT_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, f"Uzycie: /sentiment_{key} <tekst>")
            return

        text = m.group(1).strip()
        if not text:
            self.bot.reply_to(message, f"Podaj tekst po komendzie /sentiment_{key}.")
            return

        if not clf.is_trained():
            self.bot.reply_to(message, f"Model {label} nie jest wytrenowany. Uruchom /train_{key}.")
            return

        try:
            predicted = clf.predict(text)
            proba     = clf.predict_proba(text)

            proba_lines = "\n".join(
                f"  {cls}: {round(p * 100, 1)}%"
                for cls, p in sorted(proba.items(), key=lambda x: -x[1])
            )

            reply = (
                f"Analiza sentymentu ({label})\n\n"
                f"Tekst: {text}\n\n"
                f"Wynik: {predicted}\n\n"
                f"Rozklad prawdopodobienstwa:\n{proba_lines}"
            )
            self.bot.reply_to(message, reply)

        except FileNotFoundError as e:
            self.bot.reply_to(message, f"Nie mozna wczytac modelu: {e}\nUruchom /train_{key}.")
        except Exception as e:
            self.bot.reply_to(message, f"Blad klasyfikacji: {e}")

    def _handle_info(self, message, key: str):
        clf   = self.classifiers[key]
        label = self._labels[key]

        if not clf.is_trained():
            self.bot.reply_to(message, f"Model {label} nie jest wytrenowany. Uruchom /train_{key}.")
            return

        try:
            summary = clf.model_summary()
            self._send_long(message.chat.id, f"Architektura modelu {label}:\n\n{summary}")
        except Exception as e:
            self.bot.reply_to(message, f"Blad: {e}")

    def help_section(self) -> str:
        lines = ["Modele sekwencyjne\n"]
        for key, label in self._labels.items():
            clf    = self.classifiers[key]
            status = "wytrenowany" if clf.is_trained() else "brak modelu"
            lines += [
                f"{label} - {status}",
                f"  /train_{key} - trenuj i zapisz do .h5",
                f"  /sentiment_{key} <tekst> - klasyfikuj sentyment",
                f"  /{key}_info - pokaz architekture modelu",
                "",
            ]
        return "\n".join(lines)

    def _send_long(self, chat_id: int, text: str):
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            self.bot.send_message(chat_id, chunk)