import csv
import os

RESULTS_FILE = "lab2results.csv"
SIMILAR_WORDS_FILE = "lab2_similar_words.txt"

FIELDNAMES = ["dataset", "embedding", "model", "accuracy", "macro_f1", "seed"]


def save_result(row: dict):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def save_similar_words(results: dict[str, list], embedding_method: str, dataset: str):
    with open(SIMILAR_WORDS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Dataset: {dataset} | Embedding: {embedding_method}\n")
        f.write(f"{'='*60}\n")
        for word, similars in results.items():
            f.write(f"\n  [{word}]\n")
            if similars:
                for similar_word, score in similars:
                    f.write(f"    {similar_word:<20} {score:.4f}\n")
            else:
                f.write("    (brak — słowo nie w słowniku)\n")
