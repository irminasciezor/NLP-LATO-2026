# Telegram Bot

Bot do analizy tekstu w języku angielskim. Obsługuje tokenizację, lematyzację, stemming, klasyfikację i wizualizacje — wszystko przez Telegram.

---

## Struktura projektu

```
nlp_bot/
├── plots               # Przykładowe wykresy
├── bot.py              # Punkt wejścia, handlery komend
├── config.py           # Token i stałe konfiguracyjne
├── nlp_processor.py    # Operacje NLP (tokenizacja, stemming itd.)
├── visualizer.py       # Generowanie wykresów PNG
├── storage.py          # Zapis do sentences.json, statystyki zbioru
├── classifier.py       # Klasyfikator tekstu (Logistic Regression)
└── README.md
```

---

## Instalacja

###  Wklej token bota

Otwórz `config.py` i wklej token

```python
TOKEN = "TUTAJ_WKLEJ_TOKEN"
```

### Uruchom bota

```bash
python bot.py
```

## Komendy

### `/task` — pojedyncze zadanie

```
/task <zadanie> "tekst" "etykieta"
```

Dostępne zadania:

| Zadanie | Opis |
|---|---|
| `tokenize` | Podział tekstu na tokeny |
| `remove_stopwords` | Usunięcie stop-słów |
| `lemmatize` | Lematyzacja |
| `stemming` | Stemming |
| `stats` | Statystyki tekstu |
| `n-grams` | Model trigramowy z prawdopodobieństwami |
| `plot_histogram` | Histogram długości tokenów |
| `plot_wordcloud` | Chmura słów |
| `plot_bar` | Wykres najczęstszych tokenów |

Wynik operacji jest zwracany użytkownikowi, a zdanie zapisywane do `sentences.json`.

**Przykłady:**
```
/task tokenize "The quick brown fox jumps over the lazy dog" "positive"
/task stemming "The children were running and eating apples" "negative"
/task plot_wordcloud "Love joy happiness success brilliant wonderful" "positive"
```

---

### `/full_pipeline` — pełny pipeline

```
/full_pipeline "tekst" "etykieta"
```

Wykonuje kolejno: czyszczenie → tokenizację → usunięcie stop-słów → lematyzację → stemming → Bag of Words → TF-IDF → statystyki → wykresy.

Jeśli tekst zawiera więcej niż jedno zdanie, każde zdanie jest przetwarzane osobno i zapisywane z tą samą etykietą.

**Przykład:**
```
/full_pipeline "I loved this movie. The acting was superb." "positive"
```

---

### `/classifier` — klasyfikacja tekstu

```
/classifier "tekst"
/classifier with_preprocessing "tekst"
/classifier without_preprocessing "tekst"
```

Trenuje klasyfikator na danych z `sentences.json` i przewiduje klasę dla podanego tekstu. Zwraca przewidzianą etykietę wraz z procentową pewnością dla każdej klasy.

> Wymaga wcześniejszego dodania danych przez `/task` lub `/full_pipeline` — minimum kilka zdań z różnymi etykietami.

**Przykład:**
```
/classifier "This was an absolutely fantastic experience"
/classifier with_preprocessing "The product broke after one day"
```

---

### `/stats` — statystyki całego zbioru

```
/stats
```

Wczytuje wszystkie dane z `sentences.json` i zwraca:
- liczbę zdań, tokenów, unikalnych tokenów
- unikalne bigramy i trigramy (top 10)
- liczność klas
- top 10 najczęstszych słów
- wykresy: słupkowy, histogram, word cloud, liczność klas

---

## Format pliku `sentences.json`

```json
[
  { "text": "I loved this movie.", "class": "positive" },
  { "text": "The product broke after one day.", "class": "negative" },
  { "text": "The meeting is on Monday.", "class": "neutral" }
]
```
