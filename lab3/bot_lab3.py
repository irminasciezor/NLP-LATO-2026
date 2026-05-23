import telebot

from lab1.config import TOKEN
from add_sentiment_handler import AddSentimentHandler
from compare_handler import CompareHandler
from models_handler import ModelsHandler
from rnn_handler import RNNHandler
from sentiment_handler import SentimentHandler
from train_handler import TrainHandler


class NLPBotLab3:
    def __init__(self, token: str):
        self.bot = telebot.TeleBot(token)
        self.rnn           = RNNHandler(self.bot)           # /train_rnn, /train_lstm, /train_gru
        self.sentiment     = SentimentHandler(self.bot)     # /sentiment method=... text="..."
        self.train         = TrainHandler(self.bot)         # /train model=... dataset=...
        self.compare       = CompareHandler(self.bot)       # /compare dataset=... methods=...
        self.add_sentiment = AddSentimentHandler(self.bot)  # /add_sentiment "tekst" "etykieta"
        self.models        = ModelsHandler(self.bot)        # /models
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["start", "help"])
        def handle_start(message):
            self._send_help(message)

    def _send_help(self, message):
        help_text = (
            "NLP Bot — Laboratorium 3\n\n"
            + self.sentiment.help_section()
            + "\n"
            + self.train.help_section()
            + "\n"
            + self.compare.help_section()
            + "\n"
            + self.add_sentiment.help_section()
            + "\n"
            + self.models.help_section()
            + "\n"
            + self.rnn.help_section()
        )
        self.bot.reply_to(message, help_text, parse_mode="Markdown")

    def run(self):
        print("Bot Lab3 uruchomiony...")
        self.bot.polling(none_stop=True)


if __name__ == "__main__":
    NLPBotLab3(TOKEN).run()