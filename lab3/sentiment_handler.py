"""
/sentiment method=<metoda> text="tekst do analizy"

Dostępne metody:
    rule | nb | rf | transformer | textblob | stanza | simplernn | lstm | gru
"""

from __future__ import annotations

import re

from sentiment_methods import METHODS, SentimentResult, classify

_CMD_RE = re.compile(
    r'^/sentiment\s+method=(\S+)\s+text="([^"]+)"',
    re.IGNORECASE,
)


class SentimentHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["sentiment"])
        def handle_sentiment(message):
            self._handle(message)

    def _handle(self, message):
        m = _CMD_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage())
            return

        method = m.group(1).lower().strip()
        text   = m.group(2).strip()

        if method not in METHODS:
            available = ", ".join(METHODS.keys())
            self.bot.reply_to(message, f"Nieznana metoda: {method}\nDostepne: {available}")
            return

        if method in ("transformer", "stanza", "simplernn", "lstm", "gru"):
            self.bot.reply_to(message, f"Laduje model {method}...")

        try:
            result = classify(method, text)
            reply  = self._format_result(result, text)
            self.bot.reply_to(message, reply)

        except RuntimeError as e:
            self.bot.reply_to(message, f"Blad: {e}")
        except FileNotFoundError as e:
            self.bot.reply_to(message, f"Brak pliku: {e}")
        except Exception as e:
            self.bot.reply_to(message, f"Blad: {e}")

    def _format_result(self, result: SentimentResult, text: str) -> str:
        lines = [
            f"Analiza sentymentu",
            f"",
            f"Tekst: {text}",
            f"Model: {result.model_name}",
            f"",
            f"Wynik: {result.label}",
        ]

        if result.score is not None:
            score_label = self._score_label(result.method_key)
            lines.append(f"{score_label}: {result.score}")

        if result.scores:
            lines.append("")
            lines.append("Rozklad prawdopodobienstwa:")
            for cls, p in sorted(result.scores.items(), key=lambda x: -x[1]):
                lines.append(f"  {cls}: {round(p * 100, 1)}%")

        return "\n".join(lines)

    def _score_label(self, method_key: str) -> str:
        return {
            "rule":        "Dominacja slow",
            "textblob":    "Polarnosc [-1, 1]",
            "stanza":      "Sredni sentyment [0-2]",
            "transformer": "Pewnosc modelu",
        }.get(method_key, "Pewnosc")

    def _usage(self) -> str:
        methods_list = ", ".join(METHODS.keys())
        return (
            "Uzycie:\n"
            '/sentiment method=<metoda> text="tekst"\n\n'
            f"Dostepne metody: {methods_list}\n\n"
            "Przyklady:\n"
            '/sentiment method=rule text="Uwielbiam ten film!"\n'
            '/sentiment method=nb text="To byl zwykly dzien"\n'
            '/sentiment method=lstm text="Fatalny produkt, nie polecam"'
        )

    def help_section(self) -> str:
        methods_str = " | ".join(METHODS.keys())
        return (
            "---\n"
            "Analiza sentymentu\n\n"
            '/sentiment method=<metoda> text="tekst"\n\n'
            f"Metody: {methods_str}\n"
        )