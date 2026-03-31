import re
import os
import traceback

import numpy as np

from datasets import load_dataset
from embeddings import get_embedding, get_similar_words, EMBEDDING_METHODS
from models import run_experiment, resolve_models, SEEDS
from plots import (
    plot_wordcloud_corpus,
    plot_wordcloud_per_class,
    plot_confusion_matrix,
    plot_embeddings,
    plot_word_embeddings,
    save_feature_importance,
)
from results import save_result, save_similar_words

CLASSIFY_PATTERN = re.compile(
    r"dataset=(\S+)\s+method=(\S+)\s+gridsearch=(true|false)\s+run=(\d+)",
    re.IGNORECASE,
)


def parse_classify(text: str):
    m = CLASSIFY_PATTERN.search(text)
    if not m:
        return None
    return {
        "dataset":     m.group(1).lower(),
        "method":      m.group(2).lower(),
        "gridsearch":  m.group(3).lower() == "true",
        "runs":        min(int(m.group(4)), 3),
    }


def run_classify(params: dict, send_status, send_photo, send_file) -> str:
    dataset_name = params["dataset"]
    method_arg   = params["method"]
    use_gs       = params["gridsearch"]
    n_runs       = params["runs"]

    send_status(f"Ładowanie datasetu `{dataset_name}`...")
    texts, labels, class_names = load_dataset(dataset_name)
    send_status(f"Załadowano {len(texts)} dokumentów, {len(class_names)} klas.")

    # ── Word Clouds ──────────────────────────────────────────────────────────
    send_status("Generowanie word cloudów...")
    try:
        wc_path = plot_wordcloud_corpus(texts)
        send_photo(wc_path, "Word Cloud — cały korpus")
    except Exception as e:
        send_status(f"Word cloud korpusu: {e}")

    try:
        wc_paths = plot_wordcloud_per_class(texts, labels, class_names)
        for p in wc_paths[:3]:
            send_photo(p, f"Word Cloud — {os.path.basename(p)}")
    except Exception as e:
        send_status(f"Word cloud klas: {e}")

    models_to_run = resolve_models(method_arg)
    if not models_to_run:
        return f"Nieznana metoda: `{method_arg}`"

    seeds = SEEDS[:n_runs]
    all_results = []

    for emb_method in EMBEDDING_METHODS:
        send_status(f"Embedding: `{emb_method}`...")

        try:
            X, emb_model = get_embedding(emb_method, texts)
        except Exception as e:
            send_status(f"Błąd embeddingu `{emb_method}`: {e}")
            continue

        if emb_method in ("word2vec", "glove"):
            try:
                query_words = get_query_words(texts)
                similar = get_similar_words(emb_model, query_words)
                save_similar_words(similar, emb_method, dataset_name)
                send_file("lab2_similar_words.txt", f"Podobne słowa — {emb_method}")
            except Exception as e:
                send_status(f"Podobne słowa: {e}")

            try:
                query_words = get_query_words(texts)
                word_paths = plot_word_embeddings(emb_model, query_words)
                for p in word_paths:
                    send_photo(p, f"Word embedding — {emb_method} — {os.path.basename(p)}")
            except Exception as e:
                send_status(f"Wizualizacja słów: {e}")

        for model_key in models_to_run:
            send_status(f"Model: `{model_key}` | gridsearch={use_gs}")
            run_accs, run_f1s = [], []
            last_result = None

            for seed in seeds:
                try:
                    result = run_experiment(
                        X, labels, emb_method, model_key, use_gs, seed
                    )
                    run_accs.append(result["accuracy"])
                    run_f1s.append(result["macro_f1"])
                    last_result = result

                    save_result({
                        "dataset":     dataset_name,
                        "embedding":   emb_method,
                        "model":       model_key,
                        "accuracy":    round(result["accuracy"], 4),
                        "macro_f1":    round(result["macro_f1"], 4),
                        "seed":        seed,
                    })
                except Exception as e:
                    send_status(f"Błąd seed={seed}: {e}")
                    traceback.print_exc()

            if last_result is None:
                continue

            avg_acc = np.mean(run_accs)
            avg_f1  = np.mean(run_f1s)
            all_results.append({
                "embedding": emb_method,
                "model":     model_key,
                "avg_acc":   avg_acc,
                "avg_f1":    avg_f1,
            })

            # Confusion Matrix
            try:
                cm_path = plot_confusion_matrix(
                    last_result["confusion_matrix"], class_names, emb_method, model_key
                )
                send_photo(cm_path, f"Confusion Matrix — {emb_method}/{model_key}")
            except Exception as e:
                send_status(f"Confusion matrix: {e}")

            try:
                emb_paths = plot_embeddings(
                    X, labels, dataset_name, model_key, emb_method, class_names
                )
                for p in emb_paths:
                    send_photo(p, os.path.basename(p))
            except Exception as e:
                send_status(f"Wizualizacja embeddingów: {e}")

            if emb_method in ("bow", "tfidf"):
                try:
                    fi_path = save_feature_importance(
                        last_result["clf"], emb_model, model_key,
                        class_names, dataset_name, emb_method
                    )
                    if fi_path:
                        send_photo(fi_path, f"Feature Importance — {emb_method}/{model_key}")
                except Exception as e:
                    send_status(f" Feature importance: {e}")

    if os.path.exists("lab2results.csv"):
        send_file("lab2results.csv", "Wyniki eksperymentów")

    if not all_results:
        return "Żaden eksperyment nie zakończył się sukcesem."

    lines = ["*Podsumowanie eksperymentów*\n"]
    lines.append(f"Dataset: `{dataset_name}` | Runs: {n_runs} | GridSearch: {use_gs}\n")
    lines.append(f"{'Embedding':<10} {'Model':<8} {'Avg Acc':>8} {'Avg F1':>8}")
    lines.append("─" * 40)
    for r in sorted(all_results, key=lambda x: -x["avg_f1"]):
        lines.append(
            f"{r['embedding']:<10} {r['model']:<8} "
            f"{r['avg_acc']:>8.4f} {r['avg_f1']:>8.4f}"
        )

    return "\n".join(lines)
