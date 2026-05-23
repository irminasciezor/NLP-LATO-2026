# NLP Bot — Lab 2

Rozszerzenie bota z Lab 1 o klasyfikację na całych datasetach, grid search, embeddingi i wizualizacje.

---

## Struktura projektu

```
lab2/
├── bot_lab2.py           # Punkt wejścia
├── classify_handler.py   # Logika komendy /classify
├── datasets.py           # Ładowanie datasetów
├── embeddings.py         # BoW, TF-IDF, Word2Vec, GloVe
├── models.py             # Klasyfikatory, GridSearch, ewaluacja
├── plots.py              # Wszystkie wykresy
├── results.py            # Zapis do CSV i TXT
└── lab2plots/            # Generowane wykresy
```

---

## Instalacja

###  Wklej token bota

Otwórz `config.py` i wklej token

```python
TOKEN = "TUTAJ_WKLEJ_TOKEN"
```

Uruchom:
```bash
python bot_lab2.py
```

---

## Komenda `/classify`

```
/classify dataset=<dataset> method=<model> gridsearch=<true/false> run=<n>
```

### Parametry

| Parametr | Wartości | Opis |
|---|---|---|
| `dataset` | `20news_group`, `imdb`, `amazon`, `ag_news` | Zbiór danych |
| `method` | `nb`, `rf`, `mlp`, `logreg`, `all`, lub `rf,nb` | Model(e) klasyfikacji |
| `gridsearch` | `true` / `false` | Strojenie hiperparametrów |
| `run` | `1`, `2`, `3` | Liczba uruchomień (różne seedy) |

### Przykłady

```
/classify dataset=20news_group method=all gridsearch=false run=1
/classify dataset=20news_group method=logreg gridsearch=true run=2
/classify dataset=20news_group method=rf,nb gridsearch=false run=3
```

---

## Embeddingi

Bot automatycznie uruchamia wszystkie 4 metody reprezentacji tekstu:

| Metoda | Opis |
|---|---|
| `bow` | Bag of Words |
| `tfidf` | TF-IDF |
| `word2vec` | Trenowany na danych korpusu |
| `glove` | Pobierany automatycznie przez gensim (`glove-wiki-gigaword-100`) |

> Pierwsze uruchomienie z GloVe pobierze ~130MB modelu.

---

## Modele

| Skrót | Model |
|---|---|
| `nb` | Multinomial Naive Bayes |
| `rf` | Random Forest |
| `mlp` | MLPClassifier |
| `logreg` | Logistic Regression |
| `all` | Wszystkie powyższe |

---

## Generowane pliki

### Wykresy (`lab2plots/`)

| Plik | Opis |
|---|---|
| `wordcloud_corpus.png` | Word cloud całego korpusu |
| `wordcloud_class_<cls>.png` | Word cloud per klasa |
| `confusion_<emb>_<model>.png` | Macierz pomyłek |
| `<dataset>_<model>_<emb>_pca_embedding.png` | Wizualizacja PCA |
| `<dataset>_<model>_<emb>_tsne_embedding.png` | Wizualizacja t-SNE |
| `<dataset>_<model>_<emb>_svd_embedding.png` | Wizualizacja TruncatedSVD |
| `feature_importance_<emb>_<model>.png` | Feature importance (BoW/TF-IDF) |
| `word_embedding_pca.png` | Wizualizacja słów — PCA |
| `word_embedding_tsne.png` | Wizualizacja słów — t-SNE |

### Inne pliki

| Plik | Opis |
|---|---|
| `lab2results.csv` | Wyniki wszystkich eksperymentów |
| `lab2_similar_words.txt` | Podobne słowa (Word2Vec / GloVe) |

---

## Datasety z Kaggle

Dla `imdb`, `amazon`, `ag_news` pobierz CSV z Kaggle i umieść w katalogu projektu:

| Dataset | Plik | Kolumny |
|---|---|---|
| IMDB | `imdb.csv` | `review`, `sentiment` |
| Amazon | `amazon.csv` | `reviewText`, `overall` |
| AG News | `ag_news.csv` | `description`, `class` |

Dataset `20news_group` jest pobierany automatycznie przez sklearn.

---
