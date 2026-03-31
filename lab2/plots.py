import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from wordcloud import WordCloud

PLOTS_DIR = "lab2plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Word Cloud ────────────────────────────────────────────────────────────────

def plot_wordcloud_corpus(texts: list[str]) -> str:
    path = os.path.join(PLOTS_DIR, "wordcloud_corpus.png")
    _save_wordcloud(" ".join(texts), path, title="Word Cloud — cały korpus")
    return path


def plot_wordcloud_per_class(texts: list[str], labels, class_names: list[str]) -> list[str]:
    paths = []
    label_list = list(labels)
    for i, cls in enumerate(class_names):
        cls_texts = [t for t, l in zip(texts, label_list) if l == i or str(l) == cls]
        if not cls_texts:
            continue
        safe = re.sub(r"[^\w]", "_", cls)
        path = os.path.join(PLOTS_DIR, f"wordcloud_class_{safe}.png")
        _save_wordcloud(" ".join(cls_texts), path, title=f"Word Cloud — klasa: {cls}")
        paths.append(path)
    return paths


def _save_wordcloud(text: str, path: str, title: str = ""):
    wc = WordCloud(width=800, height=400, background_color="white",
                   max_words=200).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          embedding: str, model_key: str) -> str:
    path = os.path.join(PLOTS_DIR, f"confusion_{embedding}_{model_key}.png")
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(n)
    short_names = [c[:15] for c in class_names]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=6)
    ax.set_ylabel("Rzeczywista klasa")
    ax.set_xlabel("Przewidziana klasa")
    ax.set_title(f"Confusion Matrix — {embedding} / {model_key}")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


# ── Embedding Visualization ───────────────────────────────────────────────────

def plot_embeddings(X, labels, dataset: str, model_key: str,
                    embedding_method: str, class_names: list[str]) -> list[str]:
    paths = []
    X_dense = X.toarray() if sp.issparse(X) else X

    if X_dense.shape[0] > 3000:
        idx = np.random.choice(X_dense.shape[0], 3000, replace=False)
        X_dense = X_dense[idx]
        labels_sub = [labels[i] for i in idx]
    else:
        labels_sub = list(labels)

    label_ids = _encode_labels(labels_sub, class_names)
    n_components = min(50, X_dense.shape[1], X_dense.shape[0])

    # PCA
    pca = PCA(n_components=2)
    try:
        X_pca = pca.fit_transform(X_dense)
        path = os.path.join(PLOTS_DIR,
            f"{dataset}_{model_key}_{embedding_method}_pca_embedding.png")
        _scatter_plot(X_pca, label_ids, class_names,
                      f"PCA — {dataset} / {model_key} / {embedding_method}", path)
        paths.append(path)
    except Exception as e:
        print(f"PCA error: {e}")

    # t-SNE
    try:
        perp = min(30, X_dense.shape[0] - 1)
        X_tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X_dense)
        path = os.path.join(PLOTS_DIR,
            f"{dataset}_{model_key}_{embedding_method}_tsne_embedding.png")
        _scatter_plot(X_tsne, label_ids, class_names,
                      f"t-SNE — {dataset} / {model_key} / {embedding_method}", path)
        paths.append(path)
    except Exception as e:
        print(f"t-SNE error: {e}")

    # TruncatedSVD
    try:
        n_comp = min(2, X_dense.shape[1] - 1, X_dense.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_svd = svd.fit_transform(X_dense if sp.issparse(X) else sp.csr_matrix(X_dense))
        if X_svd.shape[1] >= 2:
            path = os.path.join(PLOTS_DIR,
                f"{dataset}_{model_key}_{embedding_method}_svd_embedding.png")
            _scatter_plot(X_svd[:, :2], label_ids, class_names,
                          f"TruncatedSVD — {dataset} / {model_key} / {embedding_method}", path)
            paths.append(path)
    except Exception as e:
        print(f"SVD error: {e}")

    return paths


def _encode_labels(labels, class_names: list[str]) -> list[int]:
    mapping = {c: i for i, c in enumerate(class_names)}
    result = []
    for l in labels:
        if l in mapping:
            result.append(mapping[l])
        elif isinstance(l, int) and l < len(class_names):
            result.append(l)
        else:
            result.append(0)
    return result


def _scatter_plot(X2d, label_ids, class_names, title, path):
    cmap = plt.cm.get_cmap("tab20", len(class_names))
    plt.figure(figsize=(10, 7))
    for i, cls in enumerate(class_names):
        mask = [j for j, l in enumerate(label_ids) if l == i]
        if mask:
            plt.scatter(X2d[mask, 0], X2d[mask, 1],
                        c=[cmap(i)], label=cls[:20], s=8, alpha=0.6)
    plt.title(title, fontsize=11)
    plt.legend(loc="best", markerscale=2, fontsize=6,
               ncol=2, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ── Word Embedding Visualization ──────────────────────────────────────────────

def plot_word_embeddings(model, words: list[str]) -> list[str]:
    paths = []
    wv = model.wv if hasattr(model, "wv") else model
    valid = [w for w in words if w in wv]
    if len(valid) < 2:
        return paths

    vectors = np.array([wv[w] for w in valid])

    # PCA
    try:
        X_pca = PCA(n_components=2).fit_transform(vectors)
        path = os.path.join(PLOTS_DIR, "word_embedding_pca.png")
        _word_scatter(X_pca, valid, "Word Embedding — PCA", path)
        paths.append(path)
    except Exception as e:
        print(f"Word PCA error: {e}")

    # t-SNE
    try:
        perp = min(5, len(valid) - 1)
        X_tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(vectors)
        path = os.path.join(PLOTS_DIR, "word_embedding_tsne.png")
        _word_scatter(X_tsne, valid, "Word Embedding — t-SNE", path)
        paths.append(path)
    except Exception as e:
        print(f"Word t-SNE error: {e}")

    return paths


def _word_scatter(X2d, words, title, path):
    plt.figure(figsize=(10, 7))
    plt.scatter(X2d[:, 0], X2d[:, 1], s=60, color="steelblue")
    for i, word in enumerate(words):
        plt.annotate(word, (X2d[i, 0], X2d[i, 1]), fontsize=10,
                     xytext=(5, 5), textcoords="offset points")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ── Feature Importance ────────────────────────────────────────────────────────

def save_feature_importance(clf, vectorizer, model_key: str,
                             class_names: list[str], dataset: str,
                             embedding: str, top_n: int = 10) -> str:
    path = os.path.join(PLOTS_DIR, f"feature_importance_{embedding}_{model_key}.png")
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        return ""

    try:
        if model_key == "nb":
            _plot_nb_importance(clf, feature_names, class_names, top_n, path)
        elif model_key == "rf":
            _plot_rf_importance(clf, feature_names, top_n, path)
        elif model_key == "logreg":
            _plot_logreg_importance(clf, feature_names, class_names, top_n, path)
        else:
            return ""
        return path
    except Exception as e:
        print(f"Feature importance error: {e}")
        return ""


def _plot_rf_importance(clf, feature_names, top_n, path):
    importances = clf.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 5))
    plt.barh([feature_names[i] for i in idx], importances[idx], color="steelblue")
    plt.title(f"Feature Importance (RF) — top {top_n}")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def _plot_nb_importance(clf, feature_names, class_names, top_n, path):
    fig, axes = plt.subplots(1, min(3, len(class_names)),
                             figsize=(15, 5), sharey=False)
    if len(class_names) == 1:
        axes = [axes]
    for i, (ax, cls) in enumerate(zip(axes, class_names[:3])):
        log_probs = clf.feature_log_prob_[i]
        idx = np.argsort(log_probs)[-top_n:]
        ax.barh([feature_names[j] for j in idx], log_probs[idx], color="coral")
        ax.set_title(f"NB — {cls[:15]}", fontsize=9)
    plt.suptitle(f"Top {top_n} cech (Naive Bayes)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def _plot_logreg_importance(clf, feature_names, class_names, top_n, path):
    n_show = min(3, len(class_names))
    fig, axes = plt.subplots(1, n_show, figsize=(15, 5))
    if n_show == 1:
        axes = [axes]
    for i, (ax, cls) in enumerate(zip(axes, class_names[:n_show])):
        coef = clf.coef_[i] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        idx = np.argsort(np.abs(coef))[-top_n:]
        ax.barh([feature_names[j] for j in idx], coef[idx], color="mediumseagreen")
        ax.set_title(f"LogReg — {cls[:15]}", fontsize=9)
    plt.suptitle(f"Top {top_n} cech (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
