"""
/language_detect text="tekst"

Wykrywa język tekstu przez langdetect.
"""
from __future__ import annotations
import os, re
import pandas as pd
from langdetect import detect, detect_langs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "lab4results")

LANG_NAMES = {
    "pl": "Polski", "en": "English", "de": "Deutsch",
    "fr": "Français", "es": "Español", "it": "Italiano",
    "ru": "Русский", "uk": "Українська", "cs": "Čeština",
    "nl": "Nederlands", "pt": "Português", "zh-cn": "中文",
    "ja": "日本語", "ko": "한국어", "ar": "العربية",
}

_CMD_RE = re.compile(r'^/language_detect\s+text="([^"]+)"', re.IGNORECASE)


def detect_language(text: str) -> dict:
    try:
        best   = detect(text)
        probs  = detect_langs(text)
        scores = {str(p).split(":")[0]: round(float(str(p).split(":")[1]), 4)
                  for p in probs}
        return {"lang": best, "scores": scores}
    except Exception as e:
        return {"lang": "unknown", "scores": {}, "error": str(e)}


class LanguageHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["language_detect"])
        def h(msg): self._handle(msg)

    def _handle(self, message):
        m = _CMD_RE.match(message.text)
        if not m:
            self.bot.reply_to(
                message,
                'Użycie:\n`/language_detect text="tekst"`',
                parse_mode="Markdown",
            )
            return

        text = m.group(1).strip()

        try:
            result = detect_language(text)
            lang   = result["lang"]
            name   = LANG_NAMES.get(lang, lang)
            scores = result.get("scores", {})

            lines = [
                f"Wykryty język: `{lang}` — {name}",
                "",
                "Rozkład prawdopodobieństwa:",
            ]
            for l, s in sorted(scores.items(), key=lambda x: -x[1])[:5]:
                lname = LANG_NAMES.get(l, l)
                bar   = "█" *  round(s * 10) + "░"  (10 - round(s * 10))
                lines.append(f"  `{l}` ({lname}): {bar} {round(s*100,1)}%")

            _save(text, lang, scores)
            self.bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")

        except Exception as e:
            self.bot.reply_to(message, f"Błąd: `{e}`", parse_mode="Markdown")

    def help_section(self) -> str:
        return (
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Wykrywanie języka\n"
            '`/language_detect text="tekst"`\n'
        )


def _save(text, lang, scores):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "language_detect_results.csv")
    row  = pd.DataFrame([{"text": text[:200], "detected_lang": lang,
                           "scores": str(scores)}])
    if os.path.exists(path):
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)
