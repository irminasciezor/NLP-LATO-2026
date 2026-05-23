import threading
import telebot

from classify_handler import parse_classify, run_classify
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lab1'))

from config import TOKEN


class Lab2Bot:
    def __init__(self, token: str):
        self.bot = telebot.TeleBot(token)
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["start", "help"])
        def handle_help(message):
            self.bot.reply_to(message, (
                "*NLP Bot — Lab 2*\n\n"
                "*Komenda:*\n"
                "`/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>`\n\n"
                "*Datasety:* `20news_group`, `imdb`, `amazon`, `ag_news`\n"
                "*Modele:* `nb`, `rf`, `mlp`, `logreg`, `all`\n"
                "*Embeddingi:* `bow`, `tfidf`, `word2vec`, `glove` (wszystkie uruchamiane automatycznie)\n\n"
                "*Przykłady:*\n"
                "`/classify dataset=20news_group method=all gridsearch=false run=1`\n"
                "`/classify dataset=20news_group method=logreg gridsearch=true run=2`\n"
                "`/classify dataset=20news_group method=rf,nb gridsearch=false run=3`"
            ), parse_mode="Markdown")

        @self.bot.message_handler(commands=["classify"])
        def handle_classify(message):
            threading.Thread(
                target=self._run_classify_task,
                args=(message,),
                daemon=True
            ).start()

        @self.bot.message_handler(func=lambda m: True)
        def handle_other(message):
            self.bot.reply_to(message, "Użyj `/classify` lub `/help`.", parse_mode="Markdown")

    def _run_classify_task(self, message):
        chat_id = message.chat.id

        params = parse_classify(message.text)
        if not params:
            self.bot.send_message(
                chat_id,
                "Nieprawidłowa składnia.\n"
                "Użycie: `/classify dataset=20news_group method=all gridsearch=false run=1`",
                parse_mode="Markdown"
            )
            return

        def send_status(msg: str):
            try:
                self.bot.send_message(chat_id, msg, parse_mode="Markdown")
            except Exception as e:
                print(f"send_status error: {e}")

        def send_photo(path: str, caption: str = ""):
            try:
                with open(path, "rb") as f:
                    self.bot.send_photo(chat_id, f, caption=caption)
            except Exception as e:
                print(f"send_photo error ({path}): {e}")

        def send_file(path: str, caption: str = ""):
            try:
                with open(path, "rb") as f:
                    self.bot.send_document(chat_id, f, caption=caption)
            except Exception as e:
                print(f"send_file error ({path}): {e}")

        try:
            summary = run_classify(params, send_status, send_photo, send_file)
            if len(summary) > 4000:
                for chunk in [summary[i:i+4000] for i in range(0, len(summary), 4000)]:
                    self.bot.send_message(chat_id, chunk, parse_mode="Markdown")
            else:
                self.bot.send_message(chat_id, summary, parse_mode="Markdown")
        except Exception as e:
            self.bot.send_message(chat_id, f"Błąd krytyczny: `{e}`", parse_mode="Markdown")
            import traceback
            traceback.print_exc()

    def run(self):
        print("Lab2 Bot uruchomiony...")
        self.bot.polling(none_stop=True)


if __name__ == "__main__":
    Lab2Bot(TOKEN).run()
