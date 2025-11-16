import pandas as pd, os

# Path to the sentiment report file
p = os.path.join("outputs", "news_sentiment_report.csv")

# Read CSV with correct encoding
df = pd.read_csv(p, encoding="latin1", parse_dates=["publishedAt"])

# Drop rows with missing dates or sentiment values
df = df.dropna(subset=["publishedAt", "sentiment"])

# Count unique dates
daily = df["publishedAt"].dt.date.unique()

print("\n✅ Data summary:")
print("Total articles:", len(df))
print("Distinct days:", len(daily))
print("First 5 dates:", sorted(list(daily))[:5])
print("\n✅ Sentiment distribution:\n", df["sentiment"].value_counts())
