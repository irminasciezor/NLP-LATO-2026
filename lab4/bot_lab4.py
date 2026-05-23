import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
torch.set_num_threads(1)

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_LAB3   = os.path.join(_PARENT, "lab3")

sys.path.insert(0, _PARENT)
sys.path.insert(0, _LAB3)
sys.path.insert(0, _HERE)

import telebot

try:
    from lab1.config import TOKEN
except ImportError:
    raise ImportError("Brak lab3/config.py z TOKEN.")

from lab3.add_sentiment_handler import AddSentimentHandler
from lab3.compare_handler import CompareHandler
from lab3.models_handler import ModelsHandler
from lab3.rnn_handler import RNNHandler
from lab3.sentiment_handler import SentimentHandler
from lab3.train_handler import TrainHandler

from lab4.ner_handler           import NERHandler
from lab4.nel_handler           import NELHandler
from lab4.translation_handler   import TranslationHandler
from lab4.summarization_handler import SummarizationHandler
from lab4.analyze_handler       import AnalyzeHandler
from lab4.language_handler      import LanguageHandler


class NLPBotLab4:
    def __init__(self, token: str):
        self.bot = telebot.TeleBot(token)

        self.rnn           = RNNHandler(self.bot)
        self.sentiment     = SentimentHandler(self.bot)
        self.train         = TrainHandler(self.bot)
        self.compare       = CompareHandler(self.bot)
        self.add_sentiment = AddSentimentHandler(self.bot)
        self.models        = ModelsHandler(self.bot)

        self.ner           = NERHandler(self.bot)
        self.nel           = NELHandler(self.bot)
        self.translation   = TranslationHandler(self.bot)
        self.summarization = SummarizationHandler(self.bot)
        self.analyze       = AnalyzeHandler(self.bot)
        self.language      = LanguageHandler(self.bot)

        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=["start", "help"])
        def handle_help(msg):
            self._send_help(msg)

    def _send_help(self, message):
        lab4 = (
            "*NLP Bot — Laboratorium 4*\n\n"
            + self.ner.help_section()           + "\n"
            + self.nel.help_section()           + "\n"
            + self.translation.help_section()   + "\n"
            + self.summarization.help_section() + "\n"
            + self.analyze.help_section()       + "\n"
            + self.language.help_section()
        )
        lab3 = (
            "━━━━━━━━━━━━━━━━━━━━\n*— Lab3 —*\n\n"
            + self.sentiment.help_section()     + "\n"
            + self.train.help_section()         + "\n"
            + self.compare.help_section()       + "\n"
            + self.add_sentiment.help_section() + "\n"
            + self.models.help_section()        + "\n"
            + self.rnn.help_section()
        )
        full = lab4 + "\n" + lab3
        for chunk in [full[i:i+4000] for i in range(0, len(full), 4000)]:
            self.bot.send_message(message.chat.id, chunk, parse_mode="Markdown")

    def run(self):
        print("Bot Lab4 uruchomiony...")
        self.bot.polling(none_stop=True)


if __name__ == "__main__":
    NLPBotLab4(TOKEN).run()