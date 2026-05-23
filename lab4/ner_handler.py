"""
/ner method=<spacy|stanza> text="tekst"

Instalacja:
    python -m spacy download en_core_web_sm
    python -m spacy download pl_core_news_sm
"""
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
import pandas as pd

RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4results")
VALID_METHODS = ("spacy", "stanza")
SPACY_MODELS  = {"en": "en_core_web_sm", "pl": "pl_core_news_sm"}

ENTITY_LABELS = {
    "PERSON": "PERSON", "PERSNAME": "PERSON", "persName": "PERSON",
    "ORG": "ORG", "orgName": "ORG",
    "GPE": "GPE", "placeName": "GPE", "geogName": "GPE",
    "LOC": "LOCATION", "LOCATION": "LOCATION",
    "PRODUCT": "PRODUCT",
    "DATE": "DATE", "TIME": "TIME",
    "MONEY": "MONEY", "PERCENT": "PERCENT",
    "FACILITY": "FACILITY", "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART", "MISC": "MISC",
    "NORP": "NORP", "LAW": "LAW", "LANGUAGE": "LANGUAGE",
    "CARDINAL": "CARDINAL", "ORDINAL": "ORDINAL", "QUANTITY": "QUANTITY",
}

_CMD_RE = re.compile(r'^/ner\s+method=(\S+)\s+text="([^"]+)"', re.IGNORECASE)

_spacy_cache:  dict = {}
_stanza_cache: dict = {}


@dataclass
class Entity:
    text:  str
    label: str
    start: int
    end:   int


@dataclass
class NERResult:
    method:   str
    text:     str
    entities: list[Entity] = field(default_factory=list)


def _get_spacy(lang: str = "en"):
    if lang not in _spacy_cache:
        import spacy
        name = SPACY_MODELS.get(lang, "en_core_web_sm")
        try:
            _spacy_cache[lang] = spacy.load(name)
        except OSError:
            raise RuntimeError(
                f"Model Spacy {name} nie jest zainstalowany.\n"
                f"Uruchom: python -m pip install https://github.com/explosion/spacy-models/releases/download/{name}-3.7.1/{name}-3.7.1-py3-none-any.whl"
            )
    return _spacy_cache[lang]


def _get_stanza(lang: str = "en"):
    if lang not in _stanza_cache:
        import stanza
        try:
            _stanza_cache[lang] = stanza.Pipeline(lang, processors="tokenize,ner", verbose=False)
        except Exception:
            stanza.download(lang, processors="tokenize,ner", verbose=False)
            _stanza_cache[lang] = stanza.Pipeline(lang, processors="tokenize,ner", verbose=False)
    return _stanza_cache[lang]


def run_ner(text: str, method: str, lang: str = "en") -> NERResult:
    if method == "spacy":
        nlp = _get_spacy(lang)
        doc = nlp(text)
        entities = [
            Entity(
                text=ent.text,
                label=ENTITY_LABELS.get(ent.label_, ent.label_),
                start=ent.start_char,
                end=ent.end_char,
            )
            for ent in doc.ents
        ]
    else:
        nlp = _get_stanza(lang)
        doc = nlp(text)
        entities = [
            Entity(
                text=ent.text,
                label=ENTITY_LABELS.get(ent.type, ent.type),
                start=ent.start_char,
                end=ent.end_char,
            )
            for sent in doc.sentences
            for ent in sent.ents
        ]
    return NERResult(method=method, text=text, entities=entities)


def _detect_lang(text: str) -> str:
    polish_chars = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    ratio = sum(1 for c in text if c in polish_chars) / max(len(text), 1)
    return "pl" if ratio > 0.02 else "en"


def _format_ner(result: NERResult) -> str:
    lines = [
        f"Metoda: {result.method.capitalize()}",
        f"TEXT: {result.text}",
        "",
        "ENTITIES:",
    ]
    if not result.entities:
        lines.append("Brak encji.")
    else:
        for e in result.entities:
            lines.append(f"- {e.text} ({e.label}) [{e.start}:{e.end}]")
    return "\n".join(lines)


def _save_ner(result: NERResult):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "ner_results.csv")
    rows = [
        {"method": result.method, "text": result.text,
         "entity": e.text, "label": e.label, "start": e.start, "end": e.end}
        for e in result.entities
    ] or [{"method": result.method, "text": result.text,
           "entity": "", "label": "", "start": 0, "end": 0}]
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


class NERHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["ner"])
        def h(msg): self._handle(msg)

    def _handle(self, message):
        m = _CMD_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage())
            return

        method = m.group(1).lower()
        text   = m.group(2).strip()

        if method not in VALID_METHODS:
            self.bot.reply_to(message, f"Nieznana metoda: {method}\nDostepne: spacy | stanza")
            return

        self.bot.reply_to(message, f"Laduje model {method}...")

        try:
            lang   = _detect_lang(text)
            result = run_ner(text, method, lang)
            _save_ner(result)
            self.bot.reply_to(message, _format_ner(result))
        except RuntimeError as e:
            self.bot.reply_to(message, f"Blad: {e}")
        except Exception as e:
            self.bot.reply_to(message, f"Blad NER: {e}")

    def help_section(self) -> str:
        return (
            "---\n"
            "NER - rozpoznawanie encji\n"
            '/ner method=<spacy|stanza> text="tekst"\n'
        )

    def _usage(self) -> str:
        return (
            "Uzycie:\n"
            '/ner method=<spacy|stanza> text="tekst"\n\n'
            "Przyklad:\n"
            '/ner method=spacy text="Steve Jobs founded Apple in California."'
        )