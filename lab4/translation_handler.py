"""
/translate text="tekst" target_lang=<en|pl|de|fr|es>

Auto-wykrywa język źródłowy przez langdetect.
Backend: Helsinki-NLP/Opus-MT
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "lab4results")
WORKER      = os.path.join(_HERE, "translate_worker.py")

LANG_NAMES = {
    "en": "English", "pl": "Polski", "de": "Deutsch",
    "fr": "Francais", "es": "Espanol",
}

_OPUS_MODELS: dict[str, str] = {
    "en-pl": "Helsinki-NLP/opus-mt-tc-big-en-pl",
    "pl-en": "Helsinki-NLP/opus-mt-tc-big-pl-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "pl-de": "Helsinki-NLP/opus-mt-pl-de",
    "de-pl": "Helsinki-NLP/opus-mt-de-pl",
    "pl-fr": "Helsinki-NLP/opus-mt-tc-big-pl-fr",
    "fr-pl": "Helsinki-NLP/opus-mt-tc-big-fr-pl",
    "de-fr": "Helsinki-NLP/opus-mt-de-fr",
    "fr-de": "Helsinki-NLP/opus-mt-fr-de",
    "de-es": "Helsinki-NLP/opus-mt-de-es",
    "es-de": "Helsinki-NLP/opus-mt-es-de",
    "fr-es": "Helsinki-NLP/opus-mt-fr-es",
    "es-fr": "Helsinki-NLP/opus-mt-es-fr",
}

_CMD_RE = re.compile(
    r'^/translate\s+text="([^"]+)"\s+target_lang=(\S+)',
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def translate(text: str, src: str, tgt: str) -> str:
    result = subprocess.run(
        [sys.executable, WORKER, src, tgt, text],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Blad workera tlumaczenia.")
    return result.stdout.strip()


class TranslationHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["translate"])
        def h(msg): self._handle(msg)

    def _handle(self, message):
        m = _CMD_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage())
            return

        text = m.group(1).strip()
        tgt  = m.group(2).lower().strip()

        if tgt not in LANG_NAMES:
            self.bot.reply_to(message, f"Nieznany jezyk: {tgt}\nDostepne: {' | '.join(LANG_NAMES)}")
            return

        self.bot.reply_to(message, "Wykrywam jezyk i tlumacze...")

        try:
            src = _detect_language(text)[:2].lower()
            if src not in LANG_NAMES:
                src = "en"

            if src == tgt:
                self.bot.reply_to(message, f"Wykryty jezyk ({src}) jest taki sam jak docelowy.")
                return

            pair = f"{src}-{tgt}"
            if pair not in _OPUS_MODELS:
                self.bot.reply_to(
                    message,
                    f"Para {src}->{tgt} nie jest obslugiwana.\n"
                    f"Dostepne pary: {', '.join(_OPUS_MODELS.keys())}",
                )
                return

            translated = translate(text, src, tgt)
            _save(text, translated, src, tgt)

            reply = (
                f"Source: {src} ({LANG_NAMES.get(src, src)})\n"
                f"Target: {tgt} ({LANG_NAMES[tgt]})\n\n"
                f"Translation:\n{translated}"
            )
            self.bot.reply_to(message, reply)

        except subprocess.TimeoutExpired:
            self.bot.reply_to(message, "Timeout - model tlumaczenia zbyt dlugo nie odpowiada.")
        except ValueError as e:
            self.bot.reply_to(message, f"Blad: {e}")
        except Exception as e:
            self.bot.reply_to(message, f"Blad tlumaczenia: {e}")

    def help_section(self) -> str:
        return (
            "---\n"
            "Tlumaczenie maszynowe\n"
            '/translate text="tekst" target_lang=<en|pl|de|fr|es>\n'
            "Auto-wykrywa jezyk zrodlowy. Backend: Helsinki-NLP/Opus-MT.\n"
        )

    def _usage(self) -> str:
        return (
            "Uzycie:\n"
            '/translate text="tekst" target_lang=<lang>\n\n'
            f"Jezyki docelowe: {' | '.join(LANG_NAMES)}\n\n"
            "Przyklad:\n"
            '/translate text="The quick brown fox" target_lang=pl'
        )


def _save(original, translated, src, tgt):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "translation_results.csv")
    row  = pd.DataFrame([{"src": src, "tgt": tgt, "original": original,
                           "translated": translated, "model": f"opus-mt-{src}-{tgt}"}])
    if os.path.exists(path):
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)