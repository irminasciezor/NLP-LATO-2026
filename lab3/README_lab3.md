# NLP Bot — Lab 3

Rozszerzenie bota z Lab 1 i Lab 2 o analizę sentymentu, modele sekwencyjne i porównanie metod klasyfikacji.

---

## Struktura projektu

```
lab3/
├── bot_lab3.py                  # Punkt wejścia
├── classifier.py                # Klasyfikator Logistic Regression (z Lab 2)
├── config.py                    # Token bota
├── nlp_processor.py             # Przetwarzanie tekstu
├── storage.py                   # Zapis zdań
├── visualizer.py                # Wykresy
├── sentiment_classifier.py      # Klasy SimpleRNN, LSTM, GRU (Keras)
├── sentiment_methods.py         # Wszystkie 9 metod klasyfikacji sentymentu
├── sentiment_handler.py         # Handler /sentiment
├── train_handler.py             # Handler /train
├── compare_handler.py           # Handler /compare
├── add_sentiment_handler.py     # Handler /add_sentiment
├── models_handler.py            # Handler /models
├── rnn_handler.py               # Handlery /train_rnn, /train_lstm, /train_gru
├── sentiment_dataset.csv        # Zbiór danych treningowych
├── models/                      # Wytrenowane modele (.h5 + .pkl)
└── lab3plots/                   # Generowane wykresy
```

---

## Instalacja

Wklej token bota do `config.py`:

```python
TOKEN = "TUTAJ_WKLEJ_TOKEN"
```

Zainstaluj zależności:

```bash
pip install pyTelegramBotAPI tensorflow scikit-learn pandas numpy matplotlib transformers torch textblob stanza nltk
```

Uruchom:

```bash
cd lab3
python bot_lab3.py
```

---

## Komenda `/sentiment`

```
/sentiment method=<metoda> text="tekst"
```

### Metody

| Metoda | Opis |
|---|---|
| `rule` | System regułowy (słownik słów) |
| `nb` | Naive Bayes + TF-IDF |
| `rf` | Random Forest + TF-IDF |
| `transformer` | `cardiffnlp/twitter-roberta-base-sentiment` |
| `textblob` | TextBlob (polarność [-1, 1]) |
| `stanza` | Stanza pipeline sentiment |
| `simplernn` | Sieć SimpleRNN (wymaga `/train`) |
| `lstm` | Sieć LSTM (wymaga `/train`) |
| `gru` | Sieć GRU (wymaga `/train`) |

### Przykłady

```
/sentiment method=nb text="Uwielbiam ten produkt!"
/sentiment method=lstm text="Fatalny zakup, nie polecam."
/sentiment method=transformer text="This is an amazing film."
```

---

## Komenda `/train`

```
/train model=<model> dataset=<dataset>
```

### Parametry

| Parametr | Wartości | Opis |
|---|---|---|
| `model` | `simplernn`, `lstm`, `gru` | Model sekwencyjny |
| `dataset` | `amazon`, `imdb`, `custom` | Alias datasetu |

> Wszystkie trzy aliasy korzystają z pliku `sentiment_dataset.csv`.

### Przykłady

```
/train model=lstm dataset=custom
/train model=gru dataset=imdb
/train model=simplernn dataset=amazon
```

### Generowane pliki

| Plik | Opis |
|---|---|
| `models/<model>_sentiment.h5` | Wagi modelu |
| `models/<model>_tokenizer.pkl` | Tokenizer + enkoder etykiet |
| `lab3plots/train_history_<model>_<dataset>_accuracy.png` | Wykres accuracy |
| `lab3plots/train_history_<model>_<dataset>_loss.png` | Wykres loss |

---

## Komenda `/compare`

```
/compare dataset=<dataset> methods=<metoda1,metoda2,...>
```

### Przykłady

```
/compare dataset=custom methods=rule,nb,rf
/compare dataset=imdb methods=simplernn,lstm,gru
/compare dataset=amazon methods=textblob,stanza,transformer
```

### Generowane pliki

| Plik | Opis |
|---|---|
| `lab3results.csv` | Wyniki wszystkich porównań |
| `lab3plots/compare_<metoda>_<dataset>.png` | Wykres słupkowy per metoda |

### Kolumny `lab3results.csv`

| Kolumna | Opis |
|---|---|
| `dataset` | Alias datasetu |
| `method` | Nazwa metody |
| `accuracy` | Dokładność |
| `precision` | Precyzja (macro) |
| `recall` | Czułość (macro) |
| `macro_f1` | F1 (macro) |
| `model_path` | Ścieżka do pliku modelu lub `N/A` |

---

## Komenda `/add_sentiment`

```
/add_sentiment "tekst" "etykieta"
```

Etykiety: `pozytywny` | `neutralny` | `negatywny`

```
/add_sentiment "Świetny produkt!" "pozytywny"
/add_sentiment "To był zwykły dzień." "neutralny"
```

> Tekst wielozdaniowy jest automatycznie dzielony na zdania — każde zapisywane jako osobny rekord z tą samą etykietą.

---

## Komenda `/models`

Wyświetla listę wytrenowanych modeli w katalogu `models/` wraz ze statusem tokenizera i klasami.

```
/models
```

---

## Skrócone komendy modeli sekwencyjnych

```
/train_rnn          /train_lstm          /train_gru
/sentiment_rnn      /sentiment_lstm      /sentiment_gru
/rnn_info           /lstm_info           /gru_info
```

---

## Modele sekwencyjne

Architektura każdego modelu:

```
Embedding → SimpleRNN / LSTM / GRU → Dense(relu) → Dense(softmax)
```

| Parametr | Wartość |
|---|---|
| `VOCAB_SIZE` | 5 000 |
| `MAX_LEN` | 30 |
| `EMBED_DIM` | 64 |
| `RNN_UNITS` | 64 |
| `EPOCHS` | 15 |
| `BATCH_SIZE` | 16 |
| `EARLY_STOPPING` | 10% epok (`patience=1`) |
