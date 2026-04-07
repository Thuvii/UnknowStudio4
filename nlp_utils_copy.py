import re
import csv
from collections import Counter
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk import pos_tag

from transformers import pipeline, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from nltk.corpus import stopwords
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    try:
        nltk.download("averaged_perceptron_tagger_eng")
    except:
        nltk.download("averaged_perceptron_tagger")

STOPWORDS = set(stopwords.words("english"))
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

# load bert model
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
        tokenizer=SENTIMENT_TOKENIZER,
        truncation=True,
        max_length=SENTIMENT_TOKENIZER.model_max_length,
    )

#clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -----
def remove_proper_nouns(words):
    try:
        tagged = pos_tag(words)
        return [w for w, tag in tagged if tag not in ("NNP", "NNPS")]
    except LookupError:
        # fallback: return original words if tagger unavailable
        return words

# -----
def get_word_frequency(text, top_n=20, remove_names=True):
    cleaned = clean_text(text)
    tokens = cleaned.split()

    if remove_names:
        tokens = remove_proper_nouns(tokens)

    words = [
        w for w in tokens
        if w not in STOPWORDS and len(w) > 2
    ]

    freq = Counter(words)
    return freq.most_common(top_n)

# -----
def chunk_text(text, max_tokens=None):
    max_tokens = max_tokens or SENTIMENT_TOKENIZER.model_max_length
    encoded = SENTIMENT_TOKENIZER(
        text,
        truncation=True,
        return_overflowing_tokens=True,
        max_length=max_tokens,
    )

    return [
        SENTIMENT_TOKENIZER.decode(input_ids, skip_special_tokens=True)
        for input_ids in encoded["input_ids"]
    ]

# ------
def analyze_sentiment(text, model):
    chunks = chunk_text(text)

    score = 0
    count = 0

    for chunk in chunks:
        result = model(
            chunk,
            truncation=True,
            max_length=SENTIMENT_TOKENIZER.model_max_length,
        )[0]

        if result["label"] == "POSITIVE":
            score += result["score"]
        else:
            score -= result["score"]

        count += 1

    return score / count if count > 0 else 0.0

# ---- not yet use
def extract_date_from_filename(filename):
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return datetime.strptime(match.group(), "%Y-%m-%d")
    return None

#cache
def load_sentiment_cache(cache_file):
    cache = {}
    try:
        with open(cache_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cache[row["filename"]] = float(row["score"])
    except FileNotFoundError:
        pass
    return cache

def save_sentiment_cache(cache_file, cache_dict):
    with open(cache_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "score"])
        for filename, score in cache_dict.items():
            writer.writerow([filename, score])



#streamlit stuff

def load_embed_model():
    if SentenceTransformer is None:
        raise ImportError("sentence_transformers is required for embedding features")
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_explainer():
    return pipeline("text2text-generation", model="google/flan-t5-small")

def embed_summaries(model, summaries):
    return model.encode(summaries, convert_to_tensor=True)

def explain_text(model, text):
    prompt = f"Explain this in simple terms: {text}"
    result = model(prompt, max_length=100)[0]["generated_text"]
    return result
