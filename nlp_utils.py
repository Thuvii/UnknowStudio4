import re
import csv
from collections import Counter
from transformers import pipeline
from datetime import datetime
from transformers import AutoTokenizer
# ----------------------------
# Load BERT sentiment model
# ----------------------------
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

# ----------------------------
# Stopwords
# ----------------------------

import nltk
from nltk.corpus import stopwords

# You'll need to download the data first
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
# STOPWORDS = {
#     "the","and","a","to","of","in","for","on","with",
#     "at","by","an","be","is","are","was","were",
#     "that","this","it","as","from","or","but",
#     "about","into","over","after","before","between",
#     "under","again","further","then","once","have","has","its","said","will","also"}

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ----------------------------
# Word frequency
# ----------------------------
def get_word_frequency(text, top_n=20):
    cleaned = clean_text(text)
    words = [
        w for w in cleaned.split()
        if w not in STOPWORDS and len(w) > 2
    ]
    freq = Counter(words)
    return freq.most_common(top_n)

# ----------------------------
# Chunk text
# ----------------------------
def chunk_text(text, max_words=400):
    words = text.split()
    return [
        " ".join(words[i:i+max_words])
        for i in range(0, len(words), max_words)
    ]

# ----------------------------
# Sentiment numeric
# ----------------------------
def analyze_sentiment(text, model):
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    tokens = tokenizer(
        text,
        truncation=False,
        return_overflowing_tokens=True,
        max_length=512
    )

    input_ids_list = tokens["input_ids"]

    score = 0
    count = 0

    for ids in input_ids_list:
        chunk_text = tokenizer.decode(ids, skip_special_tokens=True)
        result = model(chunk_text, truncation=True, max_length=512)[0]

        if result["label"] == "POSITIVE":
            score += result["score"]
        else:
            score -= result["score"]

        count += 1

    return score / count if count > 0 else 0.0

# ----------------------------
# Extract date from filename
# ----------------------------
def extract_date_from_filename(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        return datetime.strptime(match.group(), "%Y-%m-%d")
    return None

# ----------------------------
# Load cached sentiment
# ----------------------------
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

# ----------------------------
# Save sentiment cache
# ----------------------------
def save_sentiment_cache(cache_file, cache_dict):
    with open(cache_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "score"])
        for filename, score in cache_dict.items():
            writer.writerow([filename, score])
