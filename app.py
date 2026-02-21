import streamlit as st
import os
import matplotlib.pyplot as plt
from nlp_utils import (
    load_sentiment_model,
    get_word_frequency,
    analyze_sentiment,
    extract_date_from_filename,
    load_sentiment_cache,
    save_sentiment_cache
)

ARTICLE_FOLDER = "ans_articles"
CACHE_FILE = "sentiment_cache.csv"

@st.cache_resource
def get_model():
    return load_sentiment_model()

model = get_model()

st.title("News Article NLP Dashboard")

files = [f for f in os.listdir(ARTICLE_FOLDER) if f.endswith(".txt")]

if not files:
    st.warning("No articles found.")
    st.stop()

# -----------------------------------
# Load or Build Sentiment Cache
# -----------------------------------
sentiment_cache = load_sentiment_cache(CACHE_FILE)
updated = False

for file in files:
    if file not in sentiment_cache:
        with open(os.path.join(ARTICLE_FOLDER, file), "r", encoding="utf-8") as f:
            text = f.read()

        score = analyze_sentiment(text, model)
        sentiment_cache[file] = score
        updated = True

if updated:
    save_sentiment_cache(CACHE_FILE, sentiment_cache)

# -----------------------------------
# Single Article View
# -----------------------------------
selected_file = st.selectbox("Choose an article", files)

if selected_file:
    with open(os.path.join(ARTICLE_FOLDER, selected_file), "r", encoding="utf-8") as f:
        text = f.read()

    st.subheader("Word Frequency (Stopwords Removed)")
    freq_data = get_word_frequency(text)

    words = [w[0] for w in freq_data]
    counts = [w[1] for w in freq_data]

    fig, ax = plt.subplots()
    ax.bar(words, counts)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    score = sentiment_cache[selected_file]

    if score > 0.05:
        label = "Positive"
    elif score < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    st.subheader("Sentiment (BERT)")
    st.write(f"Sentiment: **{label}**")
    st.write(f"Score: {round(score, 3)}")

# -----------------------------------
# Sentiment Trend
# -----------------------------------
st.subheader("Sentiment Trend Over Time")

dates = []
scores = []

for file, score in sentiment_cache.items():
    date = extract_date_from_filename(file)
    if not date:
        continue
    dates.append(date)
    scores.append(score)

if dates:
    sorted_data = sorted(zip(dates, scores))
    dates_sorted, scores_sorted = zip(*sorted_data)

    fig2, ax2 = plt.subplots()
    ax2.plot(dates_sorted, scores_sorted)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sentiment Score")
    ax2.set_title("Sentiment Over Time")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
else:
    st.write("No valid dates found in filenames.")



# -----------------------------------
# Sentiment Distribution (All Articles)
# -----------------------------------
st.subheader("Overall Sentiment Distribution")

positive = 0
negative = 0
neutral = 0

for score in sentiment_cache.values():
    if score > 0.05:
        positive += 1
    elif score < -0.05:
        negative += 1
    else:
        neutral += 1

labels = ["Positive", "Negative", "Neutral"]
values = [positive, negative, neutral]

fig3, ax3 = plt.subplots()
ax3.bar(labels, values)
ax3.set_ylabel("Number of Articles")
ax3.set_title("Total Sentiment Distribution")
st.pyplot(fig3)



# -----------------------------------
# Overall Word Frequency (All Articles)
# -----------------------------------
st.subheader("Top Overall Word Frequency")

from collections import Counter
from nlp_utils import clean_text, STOPWORDS

overall_counter = Counter()

for file in files:
    with open(os.path.join(ARTICLE_FOLDER, file), "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_text(text)
    words = [
        w for w in cleaned.split()
        if w not in STOPWORDS and len(w) > 2
    ]

    overall_counter.update(words)

top_words = overall_counter.most_common(20)

if top_words:
    words = [w[0] for w in top_words]
    counts = [w[1] for w in top_words]

    fig4, ax4 = plt.subplots()
    ax4.bar(words, counts)
    ax4.set_title("Top 20 Words (All Articles)")
    plt.xticks(rotation=45)
    st.pyplot(fig4)
    
    
    # -----------------------------------
# Vocabulary Comparison: Positive vs Negative
# -----------------------------------
st.subheader("Vocabulary Comparison: Positive vs Negative Articles")

from collections import Counter
from nlp_utils import clean_text, STOPWORDS

positive_counter = Counter()
negative_counter = Counter()

for file, score in sentiment_cache.items():
    if score > 0.05:
        category = "positive"
    elif score < -0.05:
        category = "negative"
    else:
        continue  # skip neutral

    with open(os.path.join(ARTICLE_FOLDER, file), "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_text(text)
    words = [
        w for w in cleaned.split()
        if w not in STOPWORDS and len(w) > 2
    ]

    if category == "positive":
        positive_counter.update(words)
    else:
        negative_counter.update(words)

# Top 15 words in each group
top_positive = positive_counter.most_common(15)
top_negative = negative_counter.most_common(15)

# Plot Positive
if top_positive:
    words_pos = [w[0] for w in top_positive]
    counts_pos = [w[1] for w in top_positive]

    fig_pos, ax_pos = plt.subplots()
    ax_pos.bar(words_pos, counts_pos)
    ax_pos.set_title("Top Words in Positive Articles")
    plt.xticks(rotation=45)
    st.pyplot(fig_pos)

# Plot Negative
if top_negative:
    words_neg = [w[0] for w in top_negative]
    counts_neg = [w[1] for w in top_negative]

    fig_neg, ax_neg = plt.subplots()
    ax_neg.bar(words_neg, counts_neg)
    ax_neg.set_title("Top Words in Negative Articles")
    plt.xticks(rotation=45)
    st.pyplot(fig_neg)
    

    # -----------------------------------
# TF-IDF Comparison: Positive vs Negative
# -----------------------------------
st.subheader("TF-IDF Comparison (Positive vs Negative)")

from sklearn.feature_extraction.text import TfidfVectorizer

positive_docs = []
negative_docs = []

for file, score in sentiment_cache.items():
    with open(os.path.join(ARTICLE_FOLDER, file), "r", encoding="utf-8") as f:
        text = f.read()

    if score > 0.05:
        positive_docs.append(text)
    elif score < -0.05:
        negative_docs.append(text)

if positive_docs and negative_docs:

    # Combine for vectorization
    all_docs = positive_docs + negative_docs
    labels = ["positive"] * len(positive_docs) + ["negative"] * len(negative_docs)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 1)
    )

    tfidf_matrix = vectorizer.fit_transform(all_docs)
    feature_names = vectorizer.get_feature_names_out()

    import numpy as np

    # Separate groups
    pos_matrix = tfidf_matrix[:len(positive_docs)]
    neg_matrix = tfidf_matrix[len(positive_docs):]

    pos_mean = np.mean(pos_matrix.toarray(), axis=0)
    neg_mean = np.mean(neg_matrix.toarray(), axis=0)

    diff = pos_mean - neg_mean

    # Top positive-skew words
    top_pos_idx = diff.argsort()[-15:][::-1]
    top_neg_idx = diff.argsort()[:15]

    # Plot Positive Distinctive
    fig1, ax1 = plt.subplots()
    ax1.bar(feature_names[top_pos_idx], diff[top_pos_idx])
    ax1.set_title("TF-IDF Words More Associated with Positive Articles")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Plot Negative Distinctive
    fig2, ax2 = plt.subplots()
    ax2.bar(feature_names[top_neg_idx], abs(diff[top_neg_idx]))
    ax2.set_title("TF-IDF Words More Associated with Negative Articles")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

else:
    st.write("Not enough positive and negative articles for comparison.")
    


# -----------------------------------
# Topic Modeling per Sentiment (LDA)
# -----------------------------------
st.subheader("Topic Modeling per Sentiment")

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def show_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(", ".join(top_words))
    return topics

def run_lda(docs, title):
    if len(docs) < 5:
        st.write(f"Not enough documents for {title} topic modeling.")
        return

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=2000
    )

    doc_term_matrix = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=3,
        random_state=42
    )

    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = show_topics(lda, feature_names)

    st.write(f"### {title} Topics")
    for i, topic in enumerate(topics):
        st.write(f"Topic {i+1}: {topic}")

# Run LDA separately
run_lda(positive_docs, "Positive Articles")
run_lda(negative_docs, "Negative Articles")