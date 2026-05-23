"""
    /compare dataset=<amazon|imdb|custom> methods=<metoda1,metoda2,...>
"""

from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from sentiment_methods import METHODS, classify


_HERE       = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(_HERE, "sentiment_dataset.csv")
RESULTS_CSV = os.path.join(_HERE, "lab3results.csv")
PLOTS_DIR   = os.path.join(_HERE, "lab3plots")

VALID_DATASETS = ("amazon", "imdb", "custom")

_MODEL_PATHS: dict[str, str] = {
    "simplernn": os.path.join("models", "simple_rnn_sentiment.h5"),
    "lstm":      os.path.join("models", "lstm_sentiment.h5"),
    "gru":       os.path.join("models", "gru_sentiment.h5"),
}

_CMD_RE = re.compile(
    r'^/compare\s+dataset=(\S+)\s+methods=(\S+)',
    re.IGNORECASE,
)

_BAR_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0",
               "#00BCD4", "#FF5722", "#607D8B", "#E91E63"]


class CompareHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["compare"])
        def handle_compare(message):
            self._handle(message)

    def _handle(self, message):
        m = _CMD_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage(), parse_mode="Markdown")
            return

        dataset_key  = m.group(1).lower().strip()
        methods_raw  = m.group(2).lower().strip()

        if dataset_key not in VALID_DATASETS:
            self.bot.reply_to(
                message,
                f"Nieznany dataset: `{dataset_key}`\n"
                f"Dostępne: `{'` | `'.join(VALID_DATASETS)}`",
                parse_mode="Markdown",
            )
            return

        methods = [m.strip() for m in methods_raw.split(",") if m.strip()]
        unknown = [m for m in methods if m not in METHODS]
        if unknown:
            self.bot.reply_to(
                message,
                f"Nieznane metody: `{', '.join(unknown)}`\n"
                f"Dostępne: `{'` | `'.join(METHODS.keys())}`",
                parse_mode="Markdown",
            )
            return

        if not methods:
            self.bot.reply_to(message, self._usage(), parse_mode="Markdown")
            return

        self.bot.reply_to(
            message,
            f"Porównuję {len(methods)} metod(y) na datasecie `{dataset_key}`...\n"
            f"Metody: `{', '.join(methods)}`",
            parse_mode="Markdown",
        )

        try:
            df = self._load_dataset()
        except (FileNotFoundError, ValueError) as e:
            self.bot.reply_to(message, f"`{e}`", parse_mode="Markdown")
            return

        results = []
        for method in methods:
            self.bot.send_message(
                message.chat.id,
                f"Ewaluuję metodę `{method}`...",
                parse_mode="Markdown",
            )
            try:
                row = self._evaluate(method, dataset_key, df)
                results.append(row)
            except Exception as e:
                self.bot.send_message(
                    message.chat.id,
                    f"Błąd metody `{method}`: `{e}` — pomijam.",
                    parse_mode="Markdown",
                )

        if not results:
            self.bot.reply_to(
                message,
                "Żadna metoda nie zwróciła wyników.",
                parse_mode="Markdown",
            )
            return

        csv_path = self._save_csv(results)

        plot_paths = []
        for row in results:
            try:
                path = self._save_plot(row, dataset_key)
                plot_paths.append(path)
            except Exception as e:
                self.bot.send_message(
                    message.chat.id,
                    f"Błąd wykresu dla `{row['method']}`: `{e}`",
                    parse_mode="Markdown",
                )

        summary = self._format_summary(results, dataset_key, csv_path, plot_paths)
        self._send_long(message.chat.id, summary)

        for path in plot_paths:
            try:
                with open(path, "rb") as img:
                    self.bot.send_photo(message.chat.id, img)
            except Exception as e:
                self.bot.send_message(
                    message.chat.id,
                    f"Nie można wysłać wykresu `{path}`: `{e}`",
                    parse_mode="Markdown",
                )


    def _load_dataset(self) -> pd.DataFrame:
        if not os.path.exists(DATASET_CSV):
            raise FileNotFoundError(
                f"Nie znaleziono pliku: `{DATASET_CSV}`"
            )
        df = pd.read_csv(DATASET_CSV).dropna(subset=["text", "label"])
        if len(df) < 2:
            raise ValueError("Dataset ma za mało wierszy (minimum 2).")
        return df


    def _evaluate(self, method: str, dataset_key: str, df: pd.DataFrame) -> dict:
        y_true = df["label"].tolist()
        y_pred = []

        for text in df["text"].tolist():
            result = classify(method, text)
            y_pred.append(result.label)

        labels = sorted(set(y_true))

        accuracy  = round(accuracy_score(y_true, y_pred), 4)
        precision = round(precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ), 4)
        recall = round(recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ), 4)
        macro_f1 = round(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ), 4)

        model_path = _MODEL_PATHS.get(method, "N/A")
        if model_path != "N/A" and not os.path.exists(model_path):
            model_path = f"{model_path} (brak pliku)"

        return {
            "dataset":    dataset_key,
            "method":     method,
            "accuracy":   accuracy,
            "precision":  precision,
            "recall":     recall,
            "macro_f1":   macro_f1,
            "model_path": model_path,
        }


    def _save_csv(self, results: list[dict]) -> str:
        new_df = pd.DataFrame(results, columns=[
            "dataset", "method", "accuracy", "precision",
            "recall", "macro_f1", "model_path",
        ])

        if os.path.exists(RESULTS_CSV):
            existing = pd.read_csv(RESULTS_CSV)
            mask = ~(
                existing["dataset"].isin(new_df["dataset"]) &
                existing["method"].isin(new_df["method"])
            )
            combined = pd.concat([existing[mask], new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_csv(RESULTS_CSV, index=False)
        return RESULTS_CSV

    def _save_plot(self, row: dict, dataset_key: str) -> str:
        os.makedirs(PLOTS_DIR, exist_ok=True)

        method      = row["method"]
        metric_keys = ["accuracy", "precision", "recall", "macro_f1"]
        values      = [row[k] for k in metric_keys]
        labels      = ["Accuracy", "Precision", "Recall", "Macro F1"]
        color       = _BAR_COLORS[list(METHODS.keys()).index(method) % len(_BAR_COLORS)]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color=color, alpha=0.85, edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_ylim(0, 1.15)
        ax.set_title(
            f"Metryki — {method.upper()} | dataset: {dataset_key}",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Wartość metryki")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        path = os.path.join(PLOTS_DIR, f"compare_{method}_{dataset_key}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)

        return path


    def _format_summary(
        self,
        results: list[dict],
        dataset_key: str,
        csv_path: str,
        plot_paths: list[str],
    ) -> str:
        lines = [
            f"Porównanie metod — dataset: `{dataset_key}`\n",
            f"Liczba próbek: wyniki na całym `sentiment_dataset.csv`\n",
        ]

        sorted_results = sorted(results, key=lambda r: -r["macro_f1"])
        lines.append("Wyniki (posortowane wg Macro F1):")
        lines.append("```")
        lines.append(f"{'Metoda':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        lines.append("─" * 42)
        for r in sorted_results:
            lines.append(
                f"{r['method']:<12} "
                f"{r['accuracy']:>6.4f} "
                f"{r['precision']:>6.4f} "
                f"{r['recall']:>6.4f} "
                f"{r['macro_f1']:>6.4f}"
            )
        lines.append("```")

        best = sorted_results[0]
        lines.append(f"\nNajlepsza metoda: `{best['method']}` (F1: `{best['macro_f1']}`)")

        lines.append(f"\nWyniki zapisano: {os.path.relpath(csv_path, _HERE)}")
        lines.append("Wykresy:")
        for p in plot_paths:
            lines.append(f"  • {os.path.relpath(p, _HERE)}")

        return "\n".join(lines)

    def _usage(self) -> str:
        methods_list = " | ".join(f"`{k}`" for k in METHODS.keys())
        datasets_str = " | ".join(f"`{d}`" for d in VALID_DATASETS)
        return (
            "Użycie:\n"
            "`/compare dataset=<dataset> methods=<metoda1,metoda2,...>`\n\n"
            f"Datasety: {datasets_str}\n\n"
            f"Metody: {methods_list}\n\n"
            "Przykłady:\n"
            "`/compare dataset=custom methods=rule,nb,rf`\n"
            "`/compare dataset=imdb methods=textblob,stanza,transformer`\n"
            "`/compare dataset=amazon methods=simplernn,lstm,gru`"
        )

    def help_section(self) -> str:
        return (
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Porównanie metod\n\n"
            "`/compare dataset=<dataset> methods=<metoda1,metoda2,...>`\n\n"
            "Datasety: `amazon` | `imdb` | `custom`\n"
            "Metody: `rule` | `nb` | `rf` | `transformer` | `textblob` | "
            "`stanza` | `simplernn` | `lstm` | `gru`\n\n"
            "Zapisuje wyniki do `lab3results.csv`, wykresy do `lab3plots/`.\n"
        )

    def _send_long(self, chat_id: int, text: str):
        if len(text) > 4000:
            for chunk in [text[i:i + 4000] for i in range(0, len(text), 4000)]:
                self.bot.send_message(chat_id, chunk, parse_mode="Markdown")
        else:
            self.bot.send_message(chat_id, text, parse_mode="Markdown")