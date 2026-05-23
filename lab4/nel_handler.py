"""
/nel text="tekst" language=<en|pl>
/ned entity="nazwa" context="kontekst"

Źródła: Wikidata API + Wikipedia API
NED: ranking TF-IDF cosine similarity
"""
from __future__ import annotations
import os, re, time
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import requests

RESULTS_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4results")
WIKIDATA_API         = "https://www.wikidata.org/w/api.php"
CONFIDENCE_THRESHOLD = 0.0

_CMD_NEL_RE = re.compile(
    r'^/nel\s+text="([^"]+)"(?:\s+language=(\S+))?',
    re.IGNORECASE,
)
_CMD_NED_RE = re.compile(
    r'^/ned\s+entity="([^"]+)"(?:\s+context="([^"]+)")?',
    re.IGNORECASE,
)


@dataclass
class Candidate:
    entity_id:   str
    label:       str
    description: str
    source:      str
    score:       float = 0.0
    url:         str   = ""
    wiki_url:    str   = ""


@dataclass
class LinkedEntity:
    mention:    str
    ner_label:  str = ""
    candidates: list[Candidate] = field(default_factory=list)
    best:       Optional[Candidate] = None


def search_wikidata(entity: str, lang: str = "en", limit: int = 5) -> list[Candidate]:
    try:
        headers = {"User-Agent": "NLPBot/1.0 (nlpbot@example.com) python-requests"}
        resp = requests.get(WIKIDATA_API, params={
            "action": "wbsearchentities", "search": entity,
            "language": lang, "limit": limit, "format": "json",
        }, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        candidates = []
        for item in data.get("search", []):
            qid = item.get("id", "")
            candidates.append(Candidate(
                entity_id=qid,
                label=item.get("label", entity),
                description=item.get("description", ""),
                source="wikidata",
                url=f"https://www.wikidata.org/wiki/{qid}",
                wiki_url=_get_wikipedia_url(qid, lang),
            ))
        return candidates
    except Exception:
        return []


def _get_wikipedia_url(qid: str, lang: str = "en") -> str:
    try:
        headers = {"User-Agent": "NLPBot/1.0 (nlpbot@example.com) python-requests"}
        resp = requests.get(WIKIDATA_API, params={
            "action": "wbgetentities", "ids": qid,
            "props": "sitelinks/urls", "sitelinkfilter": f"{lang}wiki",
            "format": "json",
        }, headers=headers, timeout=6)
        entity = resp.json().get("entities", {}).get(qid, {})
        sl = entity.get("sitelinks", {}).get(f"{lang}wiki", {})
        return sl.get("url", "")
    except Exception:
        return ""


def rank_candidates(candidates: list[Candidate], context: str) -> list[Candidate]:
    if not candidates:
        return []

    if not context.strip():
        for i, c in enumerate(candidates):
            c.score = round(1.0 / (i + 1), 4)
        return candidates

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        docs   = [c.description or c.label for c in candidates]
        vec    = TfidfVectorizer(min_df=1).fit(docs + [context])
        scores = cosine_similarity(vec.transform([context]), vec.transform(docs))[0]
        for c, s in zip(candidates, scores):
            c.score = round(float(s), 4)
    except Exception:
        for i, c in enumerate(candidates):
            c.score = round(1.0 / (i + 1), 4)

    return sorted(candidates, key=lambda c: -c.score)


def link_entity(mention: str, context: str = "", lang: str = "en") -> LinkedEntity:
    candidates = search_wikidata(mention, lang="en", limit=5)
    time.sleep(0.3)
    ranked = rank_candidates(candidates, context=context)
    return LinkedEntity(
        mention=mention,
        candidates=ranked,
        best=ranked[0] if ranked else None,
    )


class NELHandler:
    def __init__(self, bot):
        self.bot = bot
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["nel"])
        def h_nel(msg): self._handle_nel(msg)

        @self.bot.message_handler(commands=["ned"])
        def h_ned(msg): self._handle_ned(msg)

    def _handle_nel(self, message):
        m = _CMD_NEL_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage_nel())
            return

        text = m.group(1).strip()
        lang = (m.group(2) or "en").lower()

        self.bot.reply_to(message, "Rozpoznaje encje i linkuje...")

        try:
            from ner_handler import run_ner, _detect_lang
            detected_lang = _detect_lang(text)
            ner_result    = run_ner(text, "spacy", detected_lang)

            if not ner_result.entities:
                self.bot.reply_to(message, "Brak encji w tekscie.")
                return

            lines = [f"NEL dla tekstu: {text[:80]}{'...' if len(text) > 80 else ''}\n"]
            saved = []

            seen: set[str] = set()
            for ent in ner_result.entities:
                if ent.text.lower() in seen or len(seen) >= 5:
                    continue
                seen.add(ent.text.lower())
                linked = link_entity(ent.text, context=text, lang=lang)
                linked.ner_label = ent.label
                saved.append(linked)
                lines.append(_format_linked(linked))
                lines.append("")

            _save_nel(saved)
            self._send_long(message.chat.id, "\n".join(lines))

        except Exception as e:
            self.bot.reply_to(message, f"Blad NEL: {e}")

    def _handle_ned(self, message):
        m = _CMD_NED_RE.match(message.text)
        if not m:
            self.bot.reply_to(message, self._usage_ned())
            return

        entity  = m.group(1).strip()
        context = (m.group(2) or "").strip()

        self.bot.reply_to(message, f"Disambiguacja encji: {entity}...")

        try:
            linked = link_entity(entity, context=context, lang="en")
            _save_nel([linked])
            self.bot.reply_to(message, _format_linked(linked))
        except Exception as e:
            self.bot.reply_to(message, f"Blad NED: {e}")

    def help_section(self) -> str:
        return (
            "---\n"
            "NEL / NED - linkowanie encji\n"
            '/nel text="tekst" language=<en|pl>\n'
            '/ned entity="nazwa" context="kontekst"\n'
        )

    def _usage_nel(self) -> str:
        return (
            "Uzycie:\n"
            '/nel text="tekst" language=<en|pl>\n\n'
            "Przyklad:\n"
            '/nel text="Steve Jobs visited Berlin." language=en'
        )

    def _usage_ned(self) -> str:
        return (
            "Uzycie:\n"
            '/ned entity="nazwa encji"\n'
            '/ned entity="nazwa encji" context="kontekst"\n\n'
            "Przyklad:\n"
            '/ned entity="Apple" context="Steve Jobs founded Apple in California."'
        )

    def _send_long(self, chat_id, text):
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            self.bot.send_message(chat_id, chunk)


def _format_linked(linked: LinkedEntity) -> str:
    label_str = f" ({linked.ner_label})" if linked.ner_label else ""
    lines = [f"Entity: {linked.mention}{label_str}", "Candidates:"]

    if not linked.candidates:
        lines.append("  Brak kandydatow.")
        return "\n".join(lines)

    for i, c in enumerate(linked.candidates[:5], 1):
        lines.append(f"{i}. {c.label} ({c.entity_id}) - {c.description[:80]}")
        if c.wiki_url:
            lines.append(f"   Wikipedia: {c.wiki_url}")
        lines.append(f"   Confidence: {c.score}")

    return "\n".join(lines)


def _save_nel(linked_list: list[LinkedEntity]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "nel_results.csv")
    rows = [
        {"mention": le.mention, "ner_label": le.ner_label,
         "entity_id": c.entity_id, "label": c.label,
         "description": c.description[:200], "score": c.score,
         "is_best": c == le.best, "wiki_url": c.wiki_url}
        for le in linked_list for c in le.candidates
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)