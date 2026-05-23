"""
/models
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional

_HERE      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")

_KNOWN_MODELS: dict[str, dict] = {
    "simple_rnn_sentiment.h5": {
        "label":     "SimpleRNN",
        "tokenizer": "simple_rnn_tokenizer.pkl",
    },
    "lstm_sentiment.h5": {
        "label":     "LSTM",
        "tokenizer": "lstm_tokenizer.pkl",
    },
    "gru_sentiment.h5": {
        "label":     "GRU",
        "tokenizer": "gru_tokenizer.pkl",
    },
}


@dataclass
class ModelInfo:
    filename:         str
    label:            str
    model_exists:     bool
    tokenizer_exists: bool
    classes:          Optional[list[str]]
    n_classes:        Optional[int]
    dataset:          Optional[str]


class ModelsHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["models"])
        def handle_models(message):
            self._handle(message)

    def _handle(self, message):
        infos = self._scan_models()

        if not any(i.model_exists for i in infos):
            self.bot.reply_to(
                message,
                f"Katalog models/ nie zawiera zadnych wytrenowanych modeli.\n\n"
                "Uzyj /train model=<model> dataset=<dataset> aby wytrenowac model.",
            )
            return

        reply = self._format(infos)
        self.bot.reply_to(message, reply)

    def _scan_models(self) -> list[ModelInfo]:
        infos = []

        for filename, meta in _KNOWN_MODELS.items():
            model_path     = os.path.join(MODELS_DIR, filename)
            tokenizer_path = os.path.join(MODELS_DIR, meta["tokenizer"])

            model_exists     = os.path.exists(model_path)
            tokenizer_exists = os.path.exists(tokenizer_path)

            classes   = None
            n_classes = None
            dataset   = "sentiment_dataset.csv" if tokenizer_exists else None

            if tokenizer_exists:
                try:
                    with open(tokenizer_path, "rb") as f:
                        _, label_index, n_cls = pickle.load(f)
                    classes   = list(label_index.values())
                    n_classes = n_cls
                except Exception:
                    pass

            infos.append(ModelInfo(
                filename=filename,
                label=meta["label"],
                model_exists=model_exists,
                tokenizer_exists=tokenizer_exists,
                classes=classes,
                n_classes=n_classes,
                dataset=dataset,
            ))

        return infos

    def _format(self, infos: list[ModelInfo]) -> str:
        lines = [f"Modele w katalogu models/\n"]

        for info in infos:
            status = "[OK]" if info.model_exists else "[BRAK]"
            lines.append(f"{status} {info.label} ({info.filename})")

            if info.model_exists:
                tok_status = "dostepny" if info.tokenizer_exists else "brak"
                lines.append(f"  Tokenizer & enkoder etykiet: {tok_status}")

                if info.dataset:
                    lines.append(f"  Dataset: {info.dataset}")

                if info.classes:
                    classes_str = ", ".join(sorted(info.classes))
                    lines.append(f"  Klasy ({info.n_classes}): {classes_str}")
            else:
                lines.append(f"  Model nie zostal jeszcze wytrenowany")
                lines.append(f"  Uzyj: /train model={info.label.lower()} dataset=custom")

            lines.append("")

        lines.append("Aby wytrenowac brakujacy model:")
        lines.append("/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>")

        return "\n".join(lines)

    def help_section(self) -> str:
        return (
            "---\n"
            "Zarzadzanie modelami\n\n"
            "/models - lista wytrenowanych modeli w models/\n"
        )