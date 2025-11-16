# inspect_csv.py
import pandas as pd
import os
path = os.path.join("outputs", "news_sentiment_report.csv")
print("Looking for:", os.path.abspath(path))
if not os.path.exists(path):
    print("ERROR: file not found:", path)
    raise SystemExit(1)

df = pd.read_csv(path)
print("Columns:", df.columns.tolist())
print("Total rows:", len(df))

# Count valid rows where both publishedAt and sentiment_score are present
valid = df.dropna(subset=["publishedAt", "sentiment_score"])
print("Rows with non-null publishedAt and sentiment_score:", len(valid))

print("\nSample rows (first 10):")
print(valid[["publishedAt", "sentiment_score"]].head(10).to_string(index=False))

# Quick diagnostics: unique dates and a few aggregated daily points
try:
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    daily = df.dropna(subset=["publishedAt","sentiment_score"]).groupby(df["publishedAt"].dt.date).agg(
        mean_sentiment=("sentiment_score","mean"), count=("sentiment_score","count")
    ).reset_index()
    print("\nDistinct daily points:", len(daily))
    print(daily.head(10).to_string(index=False))
except Exception as e:
    print("\nDiagnostics failed:", e)
