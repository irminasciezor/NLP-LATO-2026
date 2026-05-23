"""
Składnia:
    /add_sentiment "tekst" "etykieta"

Zachowanie przy tekście wielozdaniowym:
    Tekst jest dzielony na zdania (nltk.sent_tokenize) i każde zdanie
    zapisywane jest jako osobny rekord z tą samą etykietą. Podejście to
    zwiększa liczbę próbek treningowych i zachowuje atomowość rekordów
    (jeden rekord = jedno zdanie), co jest zgodne z formatem datasetu.
"""

from __future__ import annotations

import re

import nltk
import pandas as pd

DATASET_CSV    = "sentiment_dataset.csv"
VALID_LABELS   = {"pozytywny", "neutralny", "negatywny"}

_CMD_RE = re.compile(
    r'^/add_sentiment\s+"([^"]+)"\s+"([^"]+)"',
    re.IGNORECASE,
)

class AddSentimentHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["add_sentiment"])
        def handle_add(message):
            self._handle(message)

    def _handle(self, message):
        m = _CMD_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage(), parse_mode="Markdown")
            return

        text  = m.group(1).strip()
        label = m.group(2).strip().lower()

        if label not in VALID_LABELS:
            valid_str = " | ".join(f"`{l}`" for l in sorted(VALID_LABELS))
            self.bot.reply_to(
                message,
                f"Nieprawidłowa etykieta: `{label}`\n"
                f"Dostępne: {valid_str}",
                parse_mode="Markdown",
            )
            return

        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [text]

        if not sentences:
            sentences = [text]

        # Zapis do CSV
        try:
            added = self._append_to_csv(sentences, label)
        except Exception as e:
            self.bot.reply_to(
                message,
                f"Błąd zapisu do pliku:\n`{e}`",
                parse_mode="Markdown",
            )
            return

        if len(sentences) == 1:
            preview = f'`"{sentences[0]}"`'
            detail  = ""
        else:
            preview = f"_{len(sentences)} zdań:_\n" + "\n".join(
                f"  `{i+1}.` `\"{s}\"`" for i, s in enumerate(sentences)
            )
            detail = f" _(tekst wielozdaniowy — podzielony na {len(sentences)} rekordów)_"

        total = self._count_rows()

        reply = (
            f"Dodano do datasetu!{detail}\n\n"
            f"Etykieta: `{label}`\n"
            f"Zapisano: {preview}\n\n"
            f"Plik: `{DATASET_CSV}`\n"
            f"Łącznie rekordów w datasecie: `{total}`"
        )
        self.bot.reply_to(message, reply, parse_mode="Markdown")

    def _append_to_csv(self, sentences: list[str], label: str) -> int:
        new_rows = pd.DataFrame(
            [{"text": s, "label": label} for s in sentences]
        )

        if pd.io.common.file_exists(DATASET_CSV):
            existing = pd.read_csv(DATASET_CSV)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        combined.to_csv(DATASET_CSV, index=False)
        return len(sentences)

    def _count_rows(self) -> int:
        try:
            return len(pd.read_csv(DATASET_CSV))
        except Exception:
            return 0

    def _usage(self) -> str:
        valid_str = " | ".join(f"`{l}`" for l in sorted(VALID_LABELS))
        return (
            "Użycie:\n"
            '`/add_sentiment "tekst" "etykieta"`\n\n'
            f"Etykiety: {valid_str}\n\n"
            "Przykłady:\n"
            '`/add_sentiment "Uwielbiam ten produkt!" "pozytywny"`\n'
            '`/add_sentiment "To był zwykły dzień." "neutralny"`\n'
            '`/add_sentiment "Fatalny zakup, nie polecam." "negatywny"`\n\n'
            "_Tekst wielozdaniowy zostanie podzielony na zdania —_\n"
            "_każde zdanie zapisane jako osobny rekord z tą samą etykietą._"
        )

    def help_section(self) -> str:
        return (
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Dodawanie danych\n\n"
            '`/add_sentiment "tekst" "etykieta"`\n\n'
            "Etykiety: `negatywny` | `neutralny` | `pozytywny`\n\n"
            "Tekst wielozdaniowy jest dzielony na zdania — każde zapisywane osobno.\n"
        )
