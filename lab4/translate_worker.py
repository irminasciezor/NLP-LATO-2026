"""
Uruchamiany jako osobny proces przez translation_handler.py.
Przyjmuje argumenty: src tgt "tekst"
Wypisuje przetłumaczony tekst na stdout.
"""
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

src  = sys.argv[1]
tgt  = sys.argv[2]
text = sys.argv[3]

MODELS = {
    "en-pl": "Helsinki-NLP/opus-mt-tc-big-en-pl",
    "pl-en": "Helsinki-NLP/opus-mt-tc-big-pl-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "pl-de": "Helsinki-NLP/opus-mt-pl-de",
    "de-pl": "Helsinki-NLP/opus-mt-de-pl",
    "de-fr": "Helsinki-NLP/opus-mt-de-fr",
    "fr-de": "Helsinki-NLP/opus-mt-fr-de",
    "de-es": "Helsinki-NLP/opus-mt-de-es",
    "es-de": "Helsinki-NLP/opus-mt-es-de",
    "fr-es": "Helsinki-NLP/opus-mt-fr-es",
    "es-fr": "Helsinki-NLP/opus-mt-es-fr",
}

key = f"{src}-{tgt}"
model_name = MODELS.get(key)
if not model_name:
    print(f"ERROR:Brak modelu dla pary {src}-{tgt}", file=sys.stderr)
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
inputs    = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs   = model.generate(**inputs, max_length=512)
result    = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
