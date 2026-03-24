import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_utils_copy import (
    load_sentiment_model,
    get_word_frequency,
    analyze_sentiment,
    extract_date_from_filename,
    load_sentiment_cache,
    save_sentiment_cache,
    clean_text,
    remove_proper_nouns, 
    STOPWORDS
)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


NOISE_WORDS = {
    "new", "today", "year", "years", "time", "week",
    "day", "said", "also", "one", "two", "first","ann","ans"
}

from collections import Counter

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

#single article view
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






st.subheader("Overall Sentiment Distribution")

positive = 0
negative = 0
neutral = 0

for score in sentiment_cache.values():
    if score > 0.05:
        positive += 1
    elif score < -0.05:
        negative += 1


labels = ["Positive", "Negative"]
values = [positive, negative]

fig3, ax3 = plt.subplots()
ax3.bar(labels, values)
ax3.set_ylabel("Number of Articles")
ax3.set_title("Total Sentiment Distribution")
st.pyplot(fig3)




# word Frequency (all)

st.subheader("Top Overall Word Frequency")



overall_counter = Counter()

for file in files:
    with open(os.path.join(ARTICLE_FOLDER, file), "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens = remove_proper_nouns(tokens)

    words = [
        w for w in tokens
        if w not in STOPWORDS
        and w not in NOISE_WORDS
        and len(w) > 3
    ]

    overall_counter = Counter({
    w: c for w, c in overall_counter.items() if c > 2
})

top_words = overall_counter.most_common(20)

if top_words:
    words = [w[0] for w in top_words]
    counts = [w[1] for w in top_words]

    fig4, ax4 = plt.subplots()
    ax4.bar(words, counts)
    ax4.set_title("Top 20 Words (All Articles)")
    plt.xticks(rotation=45)
    st.pyplot(fig4)
    
    

# positive vs negative words
st.subheader("Vocabulary Comparison: Positive vs Negative Articles")


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
    tokens = cleaned.split()
    tokens = remove_proper_nouns(tokens)

    words = [
        w for w in tokens
        if w not in STOPWORDS
        and w not in NOISE_WORDS
        and len(w) > 3
    ]

    if category == "positive":
        positive_counter.update(words)
    else:
        negative_counter.update(words)
    positive_counter = Counter({
        w: c for w, c in positive_counter.items() if c > 2
    })
    negative_counter = Counter({
        w: c for w, c in negative_counter.items() if c > 2
    })

top_positive = positive_counter.most_common(15)
top_negative = negative_counter.most_common(15)

# plot positive
if top_positive:
    words_pos = [w[0] for w in top_positive]
    counts_pos = [w[1] for w in top_positive]

    fig_pos, ax_pos = plt.subplots()
    ax_pos.bar(words_pos, counts_pos)
    ax_pos.set_title("Top Words in Positive Articles")
    plt.xticks(rotation=45)
    st.pyplot(fig_pos)

# plot negative
if top_negative:
    words_neg = [w[0] for w in top_negative]
    counts_neg = [w[1] for w in top_negative]

    fig_neg, ax_neg = plt.subplots()
    ax_neg.bar(words_neg, counts_neg)
    ax_neg.set_title("Top Words in Negative Articles")
    plt.xticks(rotation=45)
    st.pyplot(fig_neg)
    
#TFDIF

#TFDIF preprocess clean text
def preprocess_for_tfidf(text):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens = remove_proper_nouns(tokens)

    tokens = [
        w for w in tokens
        if w not in STOPWORDS
        and w not in NOISE_WORDS
        and len(w) > 3
    ]

    return " ".join(tokens)

st.subheader("TF-IDF Comparison (Positive vs Negative)")

positive_docs = []
negative_docs = []

for file, score in sentiment_cache.items():
    with open(os.path.join(ARTICLE_FOLDER, file), "r", encoding="utf-8") as f:
        text = f.read()

    processed = preprocess_for_tfidf(text)

    if score > 0.05:
        positive_docs.append(processed)
    elif score < -0.05:
        negative_docs.append(processed)

if positive_docs and negative_docs:

    # combine for vector
    all_docs = positive_docs + negative_docs
    labels = ["positive"] * len(positive_docs) + ["negative"] * len(negative_docs)

    vectorizer = TfidfVectorizer(
        stop_words="english", 
        ngram_range=(1, 2),     
        min_df=2,               
        max_df=0.85 
    )

    tfidf_matrix = vectorizer.fit_transform(all_docs)
    feature_names = vectorizer.get_feature_names_out()



 
    pos_matrix = tfidf_matrix[:len(positive_docs)]
    neg_matrix = tfidf_matrix[len(positive_docs):]

    pos_mean = np.mean(pos_matrix.toarray(), axis=0)
    neg_mean = np.mean(neg_matrix.toarray(), axis=0)

    diff = pos_mean - neg_mean

    # top positive-skew words
    top_pos_idx = diff.argsort()[-15:][::-1]
    top_neg_idx = diff.argsort()[:15]

    # plot positive 
    fig1, ax1 = plt.subplots()
    ax1.bar(feature_names[top_pos_idx], diff[top_pos_idx])
    ax1.set_title("TF-IDF Words More Associated with Positive Articles")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # plot negative
    fig2, ax2 = plt.subplots()
    ax2.bar(feature_names[top_neg_idx], abs(diff[top_neg_idx]))
    ax2.set_title("TF-IDF Words More Associated with Negative Articles")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

else:
    st.write("Not enough positive and negative articles for comparison.")
    


#topic modeling
st.subheader("Topic Modeling per Sentiment")


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
        stop_words=list(STOPWORDS),
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

# run lda separately
run_lda(positive_docs, "Positive Articles");

run_lda(negative_docs, "Negative Articles")