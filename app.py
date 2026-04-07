import os
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from nlp_utils_copy import (
    STOPWORDS,
    analyze_sentiment,
    clean_text,
    get_word_frequency,
    load_sentiment_cache,
    load_sentiment_model,
    remove_proper_nouns,
    save_sentiment_cache,
)


NOISE_WORDS = {
    "new",
    "today",
    "year",
    "years",
    "time",
    "week",
    "day",
    "said",
    "also",
    "one",
    "two",
    "first",
    "ann",
    "ans",
    "world",
    "nuclear",
    "energy",
}

ARTICLE_FOLDERS = {
    "ANS": "ans_articles",
    "World Nuclear": os.path.join("World_Nuclear_Scraper", "articles"),
}
CACHE_FILE = "sentiment_cache.csv"


def build_article_index():
    article_index = {}

    for source_name, folder in ARTICLE_FOLDERS.items():
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            if not filename.endswith(".txt"):
                continue

            article_id = f"{source_name}::{filename}"
            article_index[article_id] = {
                "source": source_name,
                "filename": filename,
                "path": os.path.join(folder, filename),
                "label": f"[{source_name}] {filename}",
            }

    return dict(sorted(article_index.items(), key=lambda item: item[1]["label"].lower()))


def normalize_cache_keys(cache_dict, articles):
    normalized_cache = {}

    for cache_key, score in cache_dict.items():
        if cache_key in articles:
            normalized_cache[cache_key] = score
            continue

        legacy_ans_key = f"ANS::{cache_key}"
        if legacy_ans_key in articles:
            normalized_cache[legacy_ans_key] = score

    return normalized_cache


def score_to_label(score):
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"


def preprocess_tokens(text):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens = remove_proper_nouns(tokens)

    return [
        token
        for token in tokens
        if token not in STOPWORDS
        and token not in NOISE_WORDS
        and len(token) > 3
    ]


def build_theme_table(records, limit=12):
    counter = Counter()
    for record in records:
        counter.update(record["tokens"])

    rows = [
        {"Theme": word.title(), "Mentions": count}
        for word, count in counter.most_common(limit)
    ]
    return pd.DataFrame(rows)


def build_theme_table_for_frame(frame, limit=10):
    counter = Counter()
    for tokens in frame["tokens"]:
        counter.update(tokens)

    rows = [
        {"Theme": word.title(), "Mentions": count}
        for word, count in counter.most_common(limit)
    ]
    return pd.DataFrame(rows)


@st.cache_resource
def get_model():
    return load_sentiment_model()


@st.cache_data(show_spinner=False)
def load_dashboard_data():
    articles = build_article_index()
    cache = normalize_cache_keys(load_sentiment_cache(CACHE_FILE), articles)
    updated = False
    records = []

    model = get_model()

    for article_id, article in articles.items():
        with open(article["path"], "r", encoding="utf-8") as file:
            text = file.read()

        if article_id not in cache:
            cache[article_id] = analyze_sentiment(text, model)
            updated = True

        score = cache[article_id]
        tokens = preprocess_tokens(text)
        word_count = len(text.split())

        records.append(
            {
                "article_id": article_id,
                "source": article["source"],
                "filename": article["filename"],
                "label": article["label"],
                "text": text,
                "sentiment_score": score,
                "sentiment_label": score_to_label(score),
                "word_count": word_count,
                "tokens": tokens,
            }
        )

    if updated:
        save_sentiment_cache(CACHE_FILE, cache)

    return records


st.set_page_config(
    page_title="Nuclear News Story Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Nuclear News Story Dashboard")
st.caption(
    "A presentation-friendly view of what the combined ANS and World Nuclear article set is saying."
)

records = load_dashboard_data()

if not records:
    st.warning("No articles found.")
    st.stop()

df = pd.DataFrame(records)

with st.sidebar:
    st.header("View Options")
    selected_sentiments = st.multiselect(
        "Sentiment",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

filtered_df = df[df["sentiment_label"].isin(selected_sentiments)].copy()

if filtered_df.empty:
    st.warning("No articles match the selected filters.")
    st.stop()

total_articles = len(filtered_df)
positive_share = (filtered_df["sentiment_label"] == "Positive").mean() * 100
negative_share = (filtered_df["sentiment_label"] == "Negative").mean() * 100
neutral_share = (filtered_df["sentiment_label"] == "Neutral").mean() * 100
average_length = int(filtered_df["word_count"].mean())

st.markdown("### Executive Summary")
metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Articles Reviewed", f"{total_articles}")
metric_2.metric("Positive Tone", f"{positive_share:.0f}%")
metric_3.metric("Negative Tone", f"{negative_share:.0f}%")
metric_4.metric("Neutral Tone", f"{neutral_share:.0f}%")
metric_5.metric("Average Article Length", f"{average_length} words")

st.markdown("### Tone Breakdown")
sentiment_summary = (
    filtered_df.groupby(["sentiment_label"])
    .size()
    .reset_index(name="articles")
)

sentiment_chart = (
    alt.Chart(sentiment_summary)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X(
            "sentiment_label:N",
            sort=["Positive", "Neutral", "Negative"],
            title=None,
        ),
        y=alt.Y("articles:Q", title="Number of articles"),
        color=alt.Color(
            "sentiment_label:N",
            title="Tone",
            scale=alt.Scale(
                domain=["Positive", "Neutral", "Negative"],
                range=["#2E8B57", "#C9A227", "#B22222"],
            ),
        ),
        tooltip=["sentiment_label", "articles"],
    )
    .properties(height=340)
)
st.altair_chart(sentiment_chart, use_container_width=True)

st.markdown("### How Strong The Tone Is")
score_bins = pd.cut(
    filtered_df["sentiment_score"],
    bins=[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
    labels=[
        "Strongly Negative",
        "Leaning Negative",
        "Balanced / Mixed",
        "Leaning Positive",
        "Strongly Positive",
    ],
    include_lowest=True,
)
score_distribution = (
    score_bins.value_counts(sort=False)
    .rename_axis("Tone Range")
    .reset_index(name="Articles")
)

score_chart = (
    alt.Chart(score_distribution)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#6A8CAF")
    .encode(
        x=alt.X("Tone Range:N", title=None, sort=list(score_distribution["Tone Range"])),
        y=alt.Y("Articles:Q", title="Number of articles"),
        tooltip=["Tone Range", "Articles"],
    )
    .properties(height=320)
)
st.altair_chart(score_chart, use_container_width=True)

st.markdown("### Main Themes People Keep Returning To")
theme_df = build_theme_table(filtered_df.to_dict("records"), limit=12)

theme_chart = (
    alt.Chart(theme_df)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#2F6B7C")
    .encode(
        x=alt.X("Theme:N", sort="-y", title=None),
        y=alt.Y("Mentions:Q", title="How often the theme appears"),
        tooltip=["Theme", "Mentions"],
    )
    .properties(height=340)
)
st.altair_chart(theme_chart, use_container_width=True)

st.markdown("### Article Depth")
length_bins = pd.cut(
    filtered_df["word_count"],
    bins=[0, 300, 600, 1000, 2000, 100000],
    labels=[
        "Under 300",
        "300-600",
        "600-1,000",
        "1,000-2,000",
        "2,000+",
    ],
    include_lowest=True,
)
length_distribution = (
    length_bins.value_counts(sort=False)
    .rename_axis("Article Length")
    .reset_index(name="Articles")
)

length_chart = (
    alt.Chart(length_distribution)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#9C755F")
    .encode(
        x=alt.X("Article Length:N", title=None, sort=list(length_distribution["Article Length"])),
        y=alt.Y("Articles:Q", title="Number of articles"),
        tooltip=["Article Length", "Articles"],
    )
    .properties(height=320)
)
st.altair_chart(length_chart, use_container_width=True)

st.markdown("### Themes By Tone")
tone_theme_col1, tone_theme_col2 = st.columns(2)

positive_theme_df = build_theme_table_for_frame(
    filtered_df[filtered_df["sentiment_label"] == "Positive"], limit=8
)
negative_theme_df = build_theme_table_for_frame(
    filtered_df[filtered_df["sentiment_label"] == "Negative"], limit=8
)

with tone_theme_col1:
    st.markdown("#### Positive Coverage Themes")
    if not positive_theme_df.empty:
        positive_theme_chart = (
            alt.Chart(positive_theme_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#2E8B57")
            .encode(
                x=alt.X("Theme:N", sort="-y", title=None),
                y=alt.Y("Mentions:Q", title="Mentions"),
                tooltip=["Theme", "Mentions"],
            )
            .properties(height=320)
        )
        st.altair_chart(positive_theme_chart, use_container_width=True)
    else:
        st.write("No positive-theme data available.")

with tone_theme_col2:
    st.markdown("#### Negative Coverage Themes")
    if not negative_theme_df.empty:
        negative_theme_chart = (
            alt.Chart(negative_theme_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#B22222")
            .encode(
                x=alt.X("Theme:N", sort="-y", title=None),
                y=alt.Y("Mentions:Q", title="Mentions"),
                tooltip=["Theme", "Mentions"],
            )
            .properties(height=320)
        )
        st.altair_chart(negative_theme_chart, use_container_width=True)
    else:
        st.write("No negative-theme data available.")

st.markdown("### Standout Articles")
standout_col1, standout_col2 = st.columns(2)

most_positive = filtered_df.nlargest(5, "sentiment_score")[
    ["filename", "sentiment_label", "sentiment_score", "word_count"]
].copy()
most_negative = filtered_df.nsmallest(5, "sentiment_score")[
    ["filename", "sentiment_label", "sentiment_score", "word_count"]
].copy()

most_positive.columns = ["Article", "Tone", "Score", "Words"]
most_negative.columns = ["Article", "Tone", "Score", "Words"]
most_positive["Score"] = most_positive["Score"].round(3)
most_negative["Score"] = most_negative["Score"].round(3)

with standout_col1:
    st.markdown("#### Most Positive Articles")
    st.dataframe(most_positive, use_container_width=True, hide_index=True)

with standout_col2:
    st.markdown("#### Most Negative Articles")
    st.dataframe(most_negative, use_container_width=True, hide_index=True)

st.markdown("### Article Explorer")
selected_article_id = st.selectbox(
    "Choose an article to discuss",
    filtered_df["article_id"].tolist(),
    format_func=lambda article_id: filtered_df.set_index("article_id").loc[article_id, "filename"],
)

selected_row = filtered_df.set_index("article_id").loc[selected_article_id]

article_col, freq_col = st.columns([1.2, 1])

with article_col:
    st.markdown(f"#### {selected_row['filename']}")
    st.caption(
        f"Tone: {selected_row['sentiment_label']} | "
        f"Length: {int(selected_row['word_count'])} words"
    )
    st.text_area("Article text", selected_row["text"], height=340)

with freq_col:
    freq_data = get_word_frequency(selected_row["text"], top_n=10)
    freq_df = pd.DataFrame(freq_data, columns=["Word", "Count"])

    st.markdown("#### Most repeated words")
    if not freq_df.empty:
        freq_chart = (
            alt.Chart(freq_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#8C5E3C")
            .encode(
                x=alt.X("Word:N", sort="-y", title=None),
                y=alt.Y("Count:Q", title="Mentions"),
                tooltip=["Word", "Count"],
            )
            .properties(height=300)
        )
        st.altair_chart(freq_chart, use_container_width=True)
    else:
        st.write("No repeated words available for this article.")
