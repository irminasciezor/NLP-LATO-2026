# NLP Bot — Lab 4

Rozszerzenie bota z Lab 1, 2 i 3 o NER, linkowanie encji, tłumaczenie maszynowe i generowanie podsumowań.

---

## Struktura projektu

```
project/
├── lab3/                        # Wszystkie pliki z Lab 3
└── lab4/
    ├── bot_lab4.py              # Punkt wejścia (Lab 3 + Lab 4)
    ├── ner_handler.py           # Handler /ner
    ├── nel_handler.py           # Handler /nel, /ned
    ├── translation_handler.py   # Handler /translate
    ├── summarization_handler.py # Handler /summarize
    ├── analyze_handler.py       # Handler /analyze_entities, /knowledge_graph
    ├── language_handler.py      # Handler /language_detect
    └── lab4results/             # Generowane pliki CSV
        ├── ner_results.csv
        ├── nel_results.csv
        ├── translation_results.csv
        ├── summarization_results.csv
        ├── analyze_results.csv
        └── language_detect_results.csv
```

---

## Instalacja

Wklej token bota do `lab3/config.py`:

```python
TOKEN = "TUTAJ_WKLEJ_TOKEN"
```

Zainstaluj zależności:

```bash
pip install pyTelegramBotAPI spacy stanza transformers torch requests pandas scikit-learn langdetect nltk
```

### Modele Spacy

Pobierane automatycznie przy pierwszym użyciu. Można też zainstalować ręcznie:

```bash
python -m spacy download en_core_web_sm
python -m spacy download pl_core_news_sm
```

### Modele Stanza

Pobierane automatycznie przy pierwszym użyciu.

### Ollama (wymagana dla `/summarize`)

```bash
ollama serve
ollama pull SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M
```

Uruchom bota:

```bash
python lab4/bot_lab4.py
```

---

## Komenda `/ner`

```
/ner method=<spacy|stanza> text="tekst"
```

Język wykrywany automatycznie.

### Przykłady

```
/ner method=spacy text="Steve Jobs founded Apple in California."
/ner method=stanza text="Jan Kowalski pracuje w Warszawie."
```

### Obsługiwane typy encji

| Typ | Opis |
|---|---|
| `PERSON` | Osoby |
| `ORG` | Organizacje |
| `GPE` | Jednostki geopolityczne |
| `LOCATION` | Lokalizacje |
| `PRODUCT` | Produkty |
| `DATE` | Daty |
| `TIME` | Czasy |
| `MONEY` | Kwoty pieniężne |
| `PERCENT` | Wartości procentowe |
| `EVENT` | Wydarzenia |

---

## Komenda `/nel`

```
/nel text="tekst" language=<en|pl>
```

Rozpoznaje encje w tekście, a następnie linkuje je do Wikidata i Wikipedii.

```
/nel text="Steve Jobs visited Berlin in 2013." language=en
```

---

## Komenda `/ned`

```
/ned entity="nazwa" context="kontekst"
```

Disambiguacja pojedynczej encji na podstawie podobieństwa kontekstu (TF-IDF cosine similarity).

```
/ned entity="Apple" context="Steve Jobs founded Apple as a technology company."
```

---

## Komenda `/translate`

```
/translate text="tekst" target_lang=<en|pl|de|fr|es>
```

Język źródłowy wykrywany automatycznie. Backend: Helsinki-NLP/Opus-MT (lokalnie, bez API).

### Obsługiwane pary językowe

```
en ↔ pl,  en ↔ de,  en ↔ fr,  en ↔ es
pl ↔ de,  pl ↔ fr,  de ↔ fr,  de ↔ es,  fr ↔ es
```

### Przykłady

```
/translate text="The quick brown fox jumps over the lazy dog" target_lang=pl
/translate text="Cześć, jak się masz?" target_lang=en
/translate text="Bonjour tout le monde" target_lang=de
```

> Modele Opus-MT pobierane automatycznie przy pierwszym użyciu danej pary (~300 MB per model).

---

## Komenda `/summarize`

```
/summarize text="tekst" summary_type=<typ> length=<długość>
```

### Parametry

| Parametr | Wartości | Opis |
|---|---|---|
| `summary_type` | `extractive`, `abstractive`, `bullets` | Typ podsumowania |
| `length` | `short`, `medium`, `long` | Długość podsumowania |

### Przykłady

```
/summarize text="Polska to kraj w Europie Środkowej..." summary_type=abstractive length=medium
/summarize text="Polska to kraj w Europie Środkowej..." summary_type=bullets length=short
/summarize text="Polska to kraj w Europie Środkowej..." summary_type=extractive length=long
```

> Tryb `extractive` działa bez Ollama. Pozostałe typy wymagają uruchomionego `ollama serve`.

---

## Komenda `/analyze_entities`

```
/analyze_entities text="tekst" link=<true|false>
```

Łączy NER + NEL w jednej komendzie. Przy `link=true` linkuje encje do Wikidata i buduje graf wiedzy.

```
/analyze_entities text="Elon Musk posiada firmę Tesla w Austin." link=true
/analyze_entities text="Jan Kowalski pracuje w Google." link=false
```

---

## Komenda `/knowledge_graph`

```
/knowledge_graph text="tekst"
```

Buduje graf wiedzy z wykrytych encji i relacji między nimi.

```
/knowledge_graph text="Steve Jobs founded Apple in California."
```

### Wykrywane relacje

| Relacja | Słowa kluczowe |
|---|---|
| `founder` | founded, co-founded, założył |
| `owns` | owns, posiada, CEO of |
| `located-in` | located in, znajduje się w |
| `works-at` | works at, pracuje w |
| `acquired` | acquired, przejął, bought |
| `partner` | partner, współpracuje |

---

## Komenda `/language_detect`

```
/language_detect text="tekst"
```

```
/language_detect text="Bonjour, comment allez-vous?"
/language_detect text="Guten Morgen, wie geht es Ihnen?"
```
