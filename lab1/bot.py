import json
import os
import re

import nltk
import telebot

from classifier import TextClassifier
from config import TOKEN, TASK_ALIASES, VISUAL_TASKS
from nlp_processor import NLPProcessor, TextCleaner
from storage import SentenceStore, StatsAnalyzer
from visualizer import Visualizer


class TaskParser:
    PATTERN = re.compile(
        r'^/task\s+(\S+)\s+"([^"]+)"\s+"([^"]+)"',
        re.IGNORECASE,
    )

    def parse(self, text: str):
        m = self.PATTERN.match(text)
        if not m:
            return None
        return m.group(1).lower(), m.group(2), m.group(3)


class NLPBot:
    FULL_PIPELINE_PATTERN = re.compile(
        r'^/full_pipeline\s+"([^"]+)"\s+"([^"]+)"',
        re.IGNORECASE,
    )
    CLASSIFIER_PATTERN = re.compile(
        r'^/classifier(?:\s+(with_preprocessing|without_preprocessing))?\s+"([^"]+)"',
        re.IGNORECASE,
    )

    def __init__(self, token: str):
        self.bot = telebot.TeleBot(token)
        self.nlp = NLPProcessor()
        self.viz = Visualizer()
        self.store = SentenceStore()
        self.parser = TaskParser()
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["start", "help"])
        def handle_start(message):
            self._send_help(message)

        @self.bot.message_handler(commands=["task"])
        def handle_task(message):
            self._handle_task(message)

        @self.bot.message_handler(commands=["full_pipeline"])
        def handle_full_pipeline(message):
            self._handle_full_pipeline(message)

        @self.bot.message_handler(commands=["classifier"])
        def handle_classifier(message):
            self._handle_classifier(message)

        @self.bot.message_handler(commands=["stats"])
        def handle_stats(message):
            self._handle_stats(message)

    # ── /task ─────────────────────────────────

    def _handle_task(self, message):
        parsed = self.parser.parse(message.text)
        if not parsed:
            self.bot.reply_to(
                message,
                'Nieprawidłowa składnia.\nUżycie:\n`/task tokenize "Tekst" "etykieta"`',
                parse_mode="Markdown",
            )
            return

        task_raw, text, label = parsed
        task = TASK_ALIASES.get(task_raw)
        if not task:
            available = ", ".join(sorted(TASK_ALIASES.keys()))
            self.bot.reply_to(message, f"Nieznane zadanie: `{task_raw}`\nDostępne: {available}")
            return

        self.store.save(text, label)

        if task in VISUAL_TASKS:
            path = getattr(self.viz, task)(text)
            with open(path, "rb") as img:
                self.bot.send_photo(message.chat.id, img, caption=f"{task} | etykieta: {label}")
            os.remove(path)
            return

        result = self._run_nlp(task, text)
        reply = f"*Zadanie:* `{task}`\n*Etykieta:* `{label}`\n\n*Wynik:*\n```\n{result}\n```"
        self.bot.reply_to(message, reply, parse_mode="Markdown")

    def _run_nlp(self, task: str, text: str) -> str:
        fn = getattr(self.nlp, task)
        result = fn(text)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    # ── /full_pipeline ────────────────────────

    def _handle_full_pipeline(self, message):
        m = self.FULL_PIPELINE_PATTERN.match(message.text)
        if not m:
            self.bot.reply_to(
                message,
                'Nieprawidłowa składnia.\nUżycie:\n`/full_pipeline "Tekst" "etykieta"`',
                parse_mode="Markdown",
            )
            return

        raw_text, label = m.group(1), m.group(2)
        sentences = nltk.sent_tokenize(raw_text)
        for sent in sentences:
            self.store.save(sent, label)

        cleaner = TextCleaner()
        results = []
        all_tokens_clean = []

        for i, sent in enumerate(sentences, 1):
            cleaned = cleaner.clean(sent)
            tokens = self.nlp.tokenize(cleaned)
            no_sw = [t for t, _ in self.nlp.remove_stop_words(cleaned)]
            lemmas = self.nlp.lemmatize(cleaned)
            stems = self.nlp.stemming(cleaned)
            all_tokens_clean.extend(no_sw)
            results.append({
                "sentence": i,
                "original": sent,
                "cleaned": cleaned,
                "tokens": tokens,
                "no_stopwords": no_sw,
                "lemmas": lemmas,
                "stems": stems,
            })

        bow = self.nlp.bag_of_words(all_tokens_clean)
        cleaned_sentences = [cleaner.clean(s) for s in sentences]
        tfidf = self.nlp.tfidf(cleaned_sentences)
        stats = self.nlp.stats(raw_text)

        reply_lines = [f"*Full Pipeline | etykieta: {label}*\n"]
        for r in results:
            reply_lines.append(f"*Zdanie {r['sentence']}:* `{r['original']}`")
            reply_lines.append(f"  • cleaned: `{r['cleaned']}`")
            reply_lines.append(f"  • tokens: `{r['tokens']}`")
            reply_lines.append(f"  • no stopwords: `{r['no_stopwords']}`")
            reply_lines.append(f"  • lemmas: `{r['lemmas']}`")
            reply_lines.append(f"  • stems: `{r['stems']}`")
            reply_lines.append("")

        reply_lines += [
            "*Bag of Words:*",
            f"`{bow}`",
            "",
            "*TF-IDF:*",
            f"`{json.dumps(tfidf, ensure_ascii=False)}`",
            "",
            "*Statystyki:*",
            f"  • tokenów: {stats['num_tokens']}",
            f"  • słów: {stats['num_words']}",
            f"  • unikalnych: {stats['num_unique']}",
            f"  • średnia dł. słowa: {stats['avg_word_len']}",
            f"  • top 10: {stats['top_10']}",
        ]

        full_reply = "\n".join(reply_lines)
        self._send_long(message.chat.id, full_reply)

        try:
            for path in self.viz.full_pipeline_plots(raw_text):
                with open(path, "rb") as img:
                    self.bot.send_photo(message.chat.id, img)
                os.remove(path)
        except Exception as e:
            self.bot.send_message(message.chat.id, f"Błąd wykresów: {e}")

    # ── /classifier ───────────────────────────

    def _handle_classifier(self, message):
        m = self.CLASSIFIER_PATTERN.match(message.text)
        if not m:
            self.bot.reply_to(
                message,
                'Nieprawidłowa składnia.\nUżycie:\n'
                '`/classifier "Tekst"`\n'
                '`/classifier with_preprocessing "Tekst"`\n'
                '`/classifier without_preprocessing "Tekst"`',
                parse_mode="Markdown",
            )
            return

        flag = (m.group(1) or "without_preprocessing").lower()
        text = m.group(2)
        use_preprocessing = flag == "with_preprocessing"

        try:
            classifier = TextClassifier()
            n_samples = classifier.train(use_preprocessing=use_preprocessing)
            predicted = classifier.predict(text, use_preprocessing=use_preprocessing)
            proba = classifier.predict_proba(text, use_preprocessing=use_preprocessing)

            proba_lines = "\n".join(
                f"  • `{cls}`: {round(p * 100, 1)}%" for cls, p in proba.items()
            )
            mode = "z preprocessing" if use_preprocessing else "bez preprocessing"
            reply = (
                f"*Klasyfikator* ({mode})\n\n"
                f"*Tekst:* `{text}`\n"
                f"*Przewidziana klasa:* `{predicted}`\n\n"
                f"*Pewność:*\n{proba_lines}\n\n"
                f"_(model wytrenowany na {n_samples} zdaniach)_"
            )
            self.bot.reply_to(message, reply, parse_mode="Markdown")

        except (FileNotFoundError, ValueError) as e:
            self.bot.reply_to(message, f"{e}", parse_mode="Markdown")
        except Exception as e:
            self.bot.reply_to(message, f"Błąd klasyfikatora: `{e}`", parse_mode="Markdown")

    # ── /stats ────────────────────────────────

    def _handle_stats(self, message):
        try:
            analyzer = StatsAnalyzer()
            data = analyzer.analyze()

            top_bigrams = "\n".join(
                f"  • `{' '.join(bg)}` — {count}x" for bg, count in data["top_bigrams"]
            )
            top_trigrams = "\n".join(
                f"  • `{' '.join(tg)}` — {count}x" for tg, count in data["top_trigrams"]
            )
            class_lines = "\n".join(
                f"  • `{cls}`: {count} zdań" for cls, count in data["class_counts"].items()
            )
            top_words = "\n".join(
                f"  • `{w}` — {c}x" for w, c in data["word_freq"]
            )

            reply = (
                f"*Statystyki zbioru*\n\n"
                f"*Zdań łącznie:* {data['num_sentences']}\n"
                f"*Tokenów łącznie:* {data['num_tokens']}\n"
                f"*Unikalnych tokenów:* {data['num_unique_tokens']}\n"
                f"*Unikalnych bigramów:* {data['num_unique_bigrams']}\n"
                f"*Unikalnych trigramów:* {data['num_unique_trigrams']}\n\n"
                f"*Liczność klas:*\n{class_lines}\n\n"
                f"*Top 10 słów:*\n{top_words}\n\n"
                f"*2-gramy (top 10):*\n{top_bigrams}\n\n"
                f"*3-gramy (top 10):*\n{top_trigrams}\n\n"
                f"*Wszystkie unikalne tokeny:*\n`{', '.join(data['unique_tokens'])}`"
            )

            self._send_long(message.chat.id, reply)

            for path in self.viz.stats_plots(data["all_text"], data["class_counts"]):
                with open(path, "rb") as img:
                    self.bot.send_photo(message.chat.id, img)
                os.remove(path)

        except (FileNotFoundError, ValueError) as e:
            self.bot.reply_to(message, f"{e}", parse_mode="Markdown")
        except Exception as e:
            self.bot.reply_to(message, f"Błąd: `{e}`", parse_mode="Markdown")

    # ── /help ─────────────────────────────────

    def _send_help(self, message):
        help_text = (
            "*NLP Bot*\n\n"
            "*Tryb /task*\n"
            '`/task <zadanie> "tekst" "etykieta"`\n\n'
            "*Dostępne zadania:*\n"
            "• `tokenize` – tokenizacja\n"
            "• `remove_stopwords` – usuwanie stop-słów\n"
            "• `lemmatize` – lematyzacja\n"
            "• `stemming` – stemming\n"
            "• `stats` – statystyki tekstu\n"
            "• `n-grams` – model trigramowy\n"
            "• `plot_histogram` – histogram długości tokenów\n"
            "• `plot_wordcloud` – chmura słów\n"
            "• `plot_bar` – wykres najczęstszych tokenów\n\n"
            "*Tryb /full_pipeline*\n"
            '`/full_pipeline "tekst" "etykieta"`\n\n'
            "*Tryb /classifier*\n"
            '`/classifier "tekst"`\n'
            '`/classifier with_preprocessing "tekst"`\n\n'
            "*Tryb /stats*\n"
            "`/stats`\n\n"
        )
        self.bot.reply_to(message, help_text, parse_mode="Markdown")

    # ── helpers ───────────────────────────────

    def _send_long(self, chat_id: int, text: str):
        if len(text) > 4000:
            for chunk in [text[i:i + 4000] for i in range(0, len(text), 4000)]:
                self.bot.send_message(chat_id, chunk, parse_mode="Markdown")
        else:
            self.bot.send_message(chat_id, text, parse_mode="Markdown")

    def run(self):
        print("Bot uruchomiony...")
        self.bot.polling(none_stop=True)


if __name__ == "__main__":
    NLPBot(TOKEN).run()
