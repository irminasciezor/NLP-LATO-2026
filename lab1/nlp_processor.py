import re
from collections import Counter, defaultdict

import nltk
from nltk import trigrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

for resource in ["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "punkt_tab"]:
    nltk.download(resource, quiet=True)


class TextCleaner:
    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class NLPProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def remove_stop_words(self, text: str) -> list[tuple]:
        tokens = self.tokenize(text)
        tagged = nltk.pos_tag(tokens)
        return [(token, tag) for token, tag in tagged if token.lower() not in self.stop_words]

    def lemmatize(self, text: str) -> list[str]:
        tokens = self.tokenize(text)
        return [self.lemmatizer.lemmatize(t, pos="v") for t in tokens]

    def stemming(self, text: str) -> list[str]:
        tokens = self.tokenize(text)
        return [self.stemmer.stem(t) for t in tokens]

    def stats(self, text: str) -> dict:
        tokens = self.tokenize(text)
        words = [t for t in tokens if t.isalpha()]
        freq = Counter(words)
        return {
            "num_tokens": len(tokens),
            "num_words": len(words),
            "num_unique": len(set(words)),
            "avg_word_len": round(sum(len(w) for w in words) / len(words), 2) if words else 0,
            "top_10": freq.most_common(10),
        }

    def n_grams(self, text: str) -> dict:
        words = self.tokenize(text)
        tri = list(trigrams(words))
        model = defaultdict(lambda: defaultdict(float))
        for w1, w2, w3 in tri:
            model[(w1, w2)][w3] += 1
        for ctx in model:
            total = sum(model[ctx].values())
            for w3 in model[ctx]:
                model[ctx][w3] /= total
        return {f"{w1} {w2}": dict(v) for (w1, w2), v in model.items()}

    def bag_of_words(self, tokens: list[str]) -> dict:
        text = " ".join(tokens)
        vec = CountVectorizer()
        X = vec.fit_transform([text])
        return dict(zip(vec.get_feature_names_out(), X.toarray()[0].tolist()))

    def tfidf(self, sentences: list[str]) -> dict:
        vec = TfidfVectorizer()
        X = vec.fit_transform(sentences)
        terms = vec.get_feature_names_out()
        result = {}
        for i, sent in enumerate(sentences):
            result[f"sentence_{i + 1}"] = dict(zip(terms, X.toarray()[i].tolist()))
        return result
