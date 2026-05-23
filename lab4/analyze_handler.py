"""
/analyze_entities text="tekst" link=<true|false>
/knowledge_graph text="tekst"

Łączy NER + NEL i opcjonalnie buduje prosty graf wiedzy.
"""
from __future__ import annotations
import os, re, time
from dataclasses import dataclass
from typing import Optional
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4results")

_CMD_ANALYZE_RE = re.compile(
    r'^/analyze_entities\s+text="([^"]+)"(?:\s+link=(true|false))?',
    re.IGNORECASE,
)
_CMD_KG_RE = re.compile(
    r'^/knowledge_graph\s+text="([^"]+)"',
    re.IGNORECASE,
)

_RELATION_PATTERNS = [
    (re.compile(r'\b(founded?|co-?founded?|zalozyl|zalozyciel|stworzyl|jest tworca)\b', re.I), "founder"),
    (re.compile(r'\b(owns?|posiada|nalezy do|CEO of|jest wlascicielem)\b', re.I), "owns"),
    (re.compile(r'\b(located? in|znajduje sie w|w miescie|miesci sie|siedziba w|w)\b', re.I), "located-in"),
    (re.compile(r'\b(works? (at|for)|pracuje w|zatrudniony)\b', re.I), "works-at"),
    (re.compile(r'\b(partner|wspolpracuje|allied with)\b', re.I), "partner"),
    (re.compile(r'\b(acquired?|przejal|kupil|bought)\b', re.I), "acquired"),
]


@dataclass
class AnalyzedEntity:
    text:       str
    label:      str
    start:      int
    end:        int
    wikidata:   Optional[str] = None
    wiki_url:   Optional[str] = None
    confidence: float = 0.0


def _build_knowledge_graph(text: str, entities: list[AnalyzedEntity]) -> list[str]:
    triples: list[str] = []

    persons = [e for e in entities if e.label == "PERSON"]
    orgs    = [e for e in entities if e.label == "ORG"]
    places  = [e for e in entities if e.label in ("GPE", "LOCATION")]

    def _snippet(e1, e2):
        if e1.end <= e2.start:
            return text[e1.end:e2.start].lower()
        elif e2.end <= e1.start:
            return text[e2.end:e1.start].lower()
        return ""

    def _check(e1, e2, max_len=40):
        s = _snippet(e1, e2)
        if not s or len(s) > max_len:
            return None
        for pattern, rel in _RELATION_PATTERNS:
            if pattern.search(s):
                return rel
        return None

    def _check_loose(e1, e2, max_len=80):
        s = _snippet(e1, e2)
        if not s or len(s) > max_len:
            return None
        for pattern, rel in _RELATION_PATTERNS:
            if pattern.search(s):
                return rel
        return None

    for p in persons:
        for o in orgs:
            rel = _check(p, o)
            if rel:
                triples.append(f"{p.text} --{rel}--> {o.text}")

    for o in orgs:
        for pl in places:
            rel = _check_loose(o, pl)
            if rel and rel in ("located-in", "owns"):
                triples.append(f"{o.text} --located-in--> {pl.text}")

    return list(dict.fromkeys(triples))


class AnalyzeHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["analyze_entities"])
        def h_analyze(msg): self._handle_analyze(msg)

        @self.bot.message_handler(commands=["knowledge_graph"])
        def h_kg(msg): self._handle_kg(msg)

    def _handle_analyze(self, message):
        m = _CMD_ANALYZE_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage_analyze())
            return

        text    = m.group(1).strip()
        do_link = (m.group(2) or "false").lower() == "true"

        self.bot.reply_to(message, f"Analizuje encje...")

        try:
            entities = self._run_ner_nel(text, do_link)
            triples  = _build_knowledge_graph(text, entities)
            _save_analyze(text, entities, triples)
            reply = _format_analyze(text, entities, triples)
            self._send_long(message.chat.id, reply)
        except Exception as e:
            self.bot.reply_to(message, f"Blad: {e}")

    def _handle_kg(self, message):
        m = _CMD_KG_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage_kg())
            return

        text = m.group(1).strip()
        self.bot.reply_to(message, "Buduje graf wiedzy...")

        try:
            entities = self._run_ner_nel(text, do_link=True)
            triples  = _build_knowledge_graph(text, entities)

            if not triples:
                self.bot.reply_to(message, "Brak wykrytych relacji w tekscie.")
                return

            self.bot.reply_to(message, "KNOWLEDGE GRAPH:\n\n" + "\n".join(triples))

        except Exception as e:
            self.bot.reply_to(message, f"Blad: {e}")

    def _run_ner_nel(self, text: str, do_link: bool) -> list[AnalyzedEntity]:
        from ner_handler import run_ner, _detect_lang
        from nel_handler import link_entity

        lang       = _detect_lang(text)
        ner_result = run_ner(text, "spacy", lang)

        result: list[AnalyzedEntity] = []
        seen: set[str] = set()

        for ent in ner_result.entities:
            ae = AnalyzedEntity(text=ent.text, label=ent.label, start=ent.start, end=ent.end)
            if do_link and ent.text.lower() not in seen and len(seen) < 8:
                seen.add(ent.text.lower())
                linked = link_entity(ent.text, context=text, lang=lang)
                time.sleep(0.2)
                if linked.best:
                    ae.wikidata   = linked.best.entity_id
                    ae.wiki_url   = linked.best.wiki_url
                    ae.confidence = linked.best.score
            result.append(ae)

        return result

    def help_section(self) -> str:
        return (
            "---\n"
            "Analiza polaczona\n"
            '/analyze_entities text="tekst" link=<true|false>\n'
            '/knowledge_graph text="tekst"\n'
        )

    def _usage_analyze(self) -> str:
        return (
            "Uzycie:\n"
            '/analyze_entities text="tekst" link=<true|false>\n\n'
            "Przyklad:\n"
            '/analyze_entities text="Elon Musk owns Tesla in Austin." link=true'
        )

    def _usage_kg(self) -> str:
        return (
            "Uzycie:\n"
            '/knowledge_graph text="tekst"\n\n'
            "Przyklad:\n"
            '/knowledge_graph text="Steve Jobs founded Apple in California."'
        )

    def _send_long(self, chat_id, text):
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            self.bot.send_message(chat_id, chunk)


def _format_analyze(text: str, entities: list[AnalyzedEntity], triples: list[str]) -> str:
    lines = ["ENTITIES FOUND:"]

    for e in entities:
        lines.append(f"- {e.text} ({e.label}) [{e.start}:{e.end}]")
        if e.wikidata:
            lines.append(f"  Wikidata: {e.wikidata}")
        if e.wiki_url:
            lines.append(f"  Wikipedia: {e.wiki_url}")
        if not e.wikidata and not e.wiki_url:
            lines.append("  Wikidata: Not found")

    if triples:
        lines.append("")
        lines.append("KNOWLEDGE GRAPH:")
        lines += triples

    return "\n".join(lines)


def _save_analyze(text: str, entities: list[AnalyzedEntity], triples: list[str]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "analyze_results.csv")
    rows = [
        {"text": text[:200], "entity": e.text, "label": e.label,
         "start": e.start, "end": e.end, "wikidata": e.wikidata or "",
         "wiki_url": e.wiki_url or "", "confidence": e.confidence,
         "kg_triples": "; ".join(triples)}
        for e in entities
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)