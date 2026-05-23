"""
/summarize text="tekst" summary_type=<extractive|abstractive|bullets>
/summarize text="tekst" length=<short|medium|long>

Backend: Ollama (domyślny model: Bielik lub llama3)
"""
from __future__ import annotations
import os, re, time
import pandas as pd
import requests

RESULTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4results")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "phi3:latest")
OLLAMA_TIMEOUT = 120

VALID_TYPES   = ("extractive", "abstractive", "bullets")
VALID_LENGTHS = ("short", "medium", "long")

LENGTH_TOKENS = {"short": 80, "medium": 200, "long": 450}
LENGTH_HINTS  = {"short": "2-3 sentences", "medium": "5-7 sentences", "long": "10-15 sentences"}

_CMD_RE = re.compile(
    r'^/summarize\s+text="([^"]+)"'
    r'(?:\s+summary_type=(\S+))?'
    r'(?:\s+length=(\S+))?'
    r'(?:\s+model=(\S+))?',
    re.IGNORECASE,
)


def _build_prompt(text: str, sum_type: str, length: str) -> str:
    hint = LENGTH_HINTS.get(length, LENGTH_HINTS["medium"])
    if sum_type == "extractive":
        return (
            f"Extract the {hint} most important sentences from the text below. "
            f"Return only sentences from the original text, verbatim.\n\nText:\n{text}\n\nKey sentences:"
        )
    elif sum_type == "abstractive":
        return (
            f"Summarize the following text in {hint}. "
            f"Write a coherent summary in your own words.\n\nText:\n{text}\n\nSummary:"
        )
    elif sum_type == "bullets":
        n = {"short": 3, "medium": 5, "long": 8}.get(length, 5)
        return (
            f"Summarize the following text as {n} concise bullet points.\n\n"
            f"Text:\n{text}\n\nBullet points:"
        )
    return f"Summarize:\n{text}"


def _extractive_local(text: str, length: str) -> str:
    import nltk
    from collections import Counter
    n = {"short": 2, "medium": 4, "long": 7}.get(length, 4)
    sents = nltk.sent_tokenize(text)
    if len(sents) <= n:
        return text
    try:
        stop = set(nltk.corpus.stopwords.words("english") +
                   nltk.corpus.stopwords.words("polish"))
    except Exception:
        stop = set()
    words  = nltk.word_tokenize(text.lower())
    freq   = Counter(w for w in words if w.isalpha() and w not in stop)
    ranked = sorted(sents, key=lambda s: sum(freq.get(w.lower(), 0)
                    for w in nltk.word_tokenize(s)), reverse=True)[:n]
    return " ".join(s for s in sents if s in ranked)


def _ollama_available() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=4).status_code == 200
    except Exception:
        return False


def _call_ollama(prompt: str, model: str, max_tokens: int) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"num_predict": max_tokens, "temperature": 0.3}},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def count_tokens(text: str) -> int:
    return len(text.split())


class SummarizationHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["summarize"])
        def h(msg): self._handle(msg)

    def _handle(self, message):
        m = _CMD_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage())
            return

        text     = m.group(1).strip()
        sum_type = (m.group(2) or "abstractive").lower()
        length   = (m.group(3) or "medium").lower()
        model    = (m.group(4) or OLLAMA_MODEL).strip()

        if sum_type not in VALID_TYPES:
            self.bot.reply_to(message, f"Nieznany typ: {sum_type}\nDostepne: {' | '.join(VALID_TYPES)}")
            return

        if length not in VALID_LENGTHS:
            self.bot.reply_to(message, f"Nieznana dlugosc: {length}\nDostepne: {' | '.join(VALID_LENGTHS)}")
            return

        self.bot.reply_to(message, f"Generuje podsumowanie ({sum_type} / {length})...")

        try:
            t0 = time.time()
            summary, backend = self._summarize(text, sum_type, length, model)
            elapsed = round(time.time() - t0, 2)

            _save(text, summary, sum_type, length, model, backend)

            type_label   = {"extractive": "Extractive", "abstractive": "Abstractive", "bullets": "Bullet-point"}[sum_type]
            length_label = length.capitalize()
            model_label  = "local (TF-IDF)" if backend == "local" else model

            reply = (
                f"Model: {model_label}\n"
                f"Text length: {count_tokens(text)} tokens\n"
                f"Summary type: {type_label}\n"
                f"Summary length: {length_label}\n\n"
                f"SUMMARY:\n{summary}\n\n"
                f"Generation time: {elapsed}s"
            )
            self._send_long(message.chat.id, reply)

        except RuntimeError as e:
            self.bot.reply_to(message, f"Blad: {e}")
        except Exception as e:
            self.bot.reply_to(message, f"Blad: {e}")

    def _summarize(self, text, sum_type, length, model):
        if sum_type == "extractive" and not _ollama_available():
            return _extractive_local(text, length), "local"
        if not _ollama_available():
            raise RuntimeError(
                f"Ollama niedostepna ({OLLAMA_URL}).\n"
                "Uruchom: ollama serve\n"
                "Tryb extractive dziala bez Ollama."
            )
        prompt = _build_prompt(text, sum_type, length)
        return _call_ollama(prompt, model, LENGTH_TOKENS[length]), "ollama"

    def help_section(self) -> str:
        status = "online" if _ollama_available() else "offline"
        return (
            "---\n"
            f"Podsumowania (Ollama {status})\n"
            '/summarize text="tekst" summary_type=<typ> length=<dlugosc>\n'
            f"Typy: {' | '.join(VALID_TYPES)}\n"
            f"Dlugosci: {' | '.join(VALID_LENGTHS)}\n"
        )

    def _usage(self) -> str:
        return (
            "Uzycie:\n"
            '/summarize text="tekst" summary_type=<typ>\n'
            '/summarize text="tekst" summary_type=<typ> length=<dlugosc>\n\n'
            f"Typy: {' | '.join(VALID_TYPES)}\n"
            f"Dlugosci: {' | '.join(VALID_LENGTHS)}\n\n"
            "Przyklad:\n"
            '/summarize text="Polska to kraj..." summary_type=abstractive length=medium'
        )

    def _send_long(self, chat_id, text):
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            self.bot.send_message(chat_id, chunk)


def _save(original, summary, sum_type, length, model, backend):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "summarization_results.csv")
    row  = pd.DataFrame([{"type": sum_type, "length": length, "model": model,
                           "backend": backend, "original": original[:500], "summary": summary}])
    if os.path.exists(path):
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)