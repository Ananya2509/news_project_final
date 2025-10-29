#!/usr/bin/env python3
"""
news_pipeline.py
Fetch news from NewsAPI, clean text, analyze word counts, create wordcloud,
perform sentiment analysis with TextBlob, plot results and save CSV.
"""
import os 
from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from dotenv import load_dotenv 

# Load .env (looks for .env file in project root)
load_dotenv()
print("Starting program...")

# Download required NLTK data (runs the first time)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Configuration
API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    raise SystemExit("ERROR: Please set NEWS_API_KEY in .env or environment variables.")
print("API key loaded:", API_KEY[:6], "********")
BASE_URL = "https://newsapi.org/v2/everything"

# Fetching
def fetch_news(query="AI OR artificial intelligence", from_days=30, page_size=50):
    """Fetch a single page of news (page_size up to 100). Uses NewsAPI 'from' date."""
    from_date = (datetime.utcnow() - timedelta(days=from_days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "from": from_date,
        "pageSize": page_size,
        "apiKey": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("articles", [])

def articles_to_df(articles):
    rows = []
    for a in articles:
        rows.append({
            "source": (a.get("source") or {}).get("name"),
            "author": a.get("author"),
            "title": a.get("title"),
            "description": a.get("description"),
            "content": a.get("content"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
        })
    return pd.DataFrame(rows)

# Text cleaning
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)       # remove links
    text = re.sub(r"[^a-z\s]", " ", text)                    # remove special chars/numbers
    words = [w for w in text.split() if w and w not in STOPWORDS]
    return " ".join(words)

# Analysis & visualization
def plot_word_counts(df, save=False):
    df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    ax = df["word_count"].plot(kind="bar", figsize=(12, 4))
    ax.set_title("Word Count per Article")
    ax.set_xlabel("Article index")
    ax.set_ylabel("Word count")
    plt.tight_layout()
    if save:
        plt.savefig("word_count_per_article.png")
    plt.show()

def plot_top_words(df, n=10, save=False):
    all_words = " ".join(df["cleaned_text"]).split()
    counter = Counter(all_words).most_common(n)
    if not counter:
        print("No words to plot.")
        return
    words, counts = zip(*counter)
    plt.figure(figsize=(8,4))
    plt.bar(words, counts)
    plt.title(f"Top {n} Most Frequent Words")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig("top_words.png")
    plt.show()

def plot_wordcloud(df, save=False):
    text = " ".join(df["cleaned_text"])
    if not text.strip():
        print("No text for wordcloud.")
        return
    wc = WordCloud(width=900, height=450, background_color="white").generate(text)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    if save:
        wc.to_file("wordcloud.png")
    plt.show()

# Sentiment
def analyze_sentiment(text):
    if not text:
        return "neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def plot_sentiment_distribution(df, save=False):
    counts = df["sentiment"].value_counts().reindex(["positive","neutral","negative"]).fillna(0)
    counts.plot(kind="bar", figsize=(6,4))
    plt.title("Sentiment Distribution")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    if save:
        plt.savefig("sentiment_distribution.png")
    plt.show()

# Main pipeline
def main():
    print("Fetching articles...")
    articles = fetch_news(query="AI OR artificial intelligence", from_days=30, page_size=100)
    df = articles_to_df(articles)
    print(f"Collected {len(df)} articles.")
    # Merge text fields and clean
    df["raw_text"] = (df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["content"].fillna(""))
    df["cleaned_text"] = df["raw_text"].apply(clean_text)

    # Word count and visuals
    plot_word_counts(df)
    plot_top_words(df, n=10)
    plot_wordcloud(df)

    # Sentiment
    df["sentiment"] = df["cleaned_text"].apply(analyze_sentiment)
    plot_sentiment_distribution(df)

    # Save results
    print("Articles fetched:", len(df))
    out_file = "news_data_with_sentiment.csv"
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Saved final data to {out_file}")

if __name__ == "__main__":
    main()
