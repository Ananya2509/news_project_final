"""
main.py
-----------------------------------
News Sentiment Analysis using Google Gemini
-----------------------------------
1. Reads a CSV or Excel file of news headlines/articles.
2. Calls Gemini API via Google Cloud Service Account authentication.
3. Classifies each headline as positive, neutral, or negative.
4. Saves results to outputs/news_sentiment_report.csv and .xlsx.
"""

import os
import pandas as pd
import time
import json
import re
import google.auth
import google.generativeai as genai

# ---------- STEP 1: Load your data ----------
# (Change filename if needed)
DATA_FILE = "news_data_with_sentiment.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ Data file not found: {DATA_FILE}")

# Read CSV (or Excel)
if DATA_FILE.endswith(".csv"):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.read_excel(DATA_FILE)

print(f"✅ Loaded {len(df)} records from {DATA_FILE}")

# Check if there's a 'title' column
if "title" not in df.columns:
    raise KeyError("❌ Missing required column 'title' in data file!")

# ---------- STEP 2: Authenticate with Google Cloud ----------
try:
    credentials, project = google.auth.default()
    genai.configure(credentials=credentials)
    print(f"✅ Authenticated successfully with project: {project}")
except Exception as e:
    print("❌ Authentication failed:", e)
    raise SystemExit

# ---------- STEP 3: Set up Gemini model ----------
# Confirm available models using check_models.py before running
model = genai.GenerativeModel("models/gemini-2.5-flash")

print("✅ Gemini model initialized: models/gemini-2.5-flash")

# ---------- Helper functions ----------
def call_gemini(prompt):
    """Calls Gemini model with a given text prompt."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini call error:", e)
        return None


def safe_parse_json(text):
    """Parses JSON safely from Gemini output."""
    if not text:
        return {"label": "neutral", "score": 0.0}
    text = text.replace("json", "").replace("", "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {"label": "neutral", "score": 0.0}


# ---------- STEP 4: Run sentiment classification ----------
labels, scores = [], []

print("🔄 Starting sentiment analysis...\n")

for i, row in df.iterrows():
    headline = str(row.get("title", "")).strip()

    if len(headline) < 5:
        labels.append("neutral")
        scores.append(0.0)
        continue

    prompt = (
        "You are a sentiment classifier. Respond ONLY in JSON like "
        '{"label":"positive|neutral|negative","score":-1.0..1.0}. '
        f'Headline: \"{headline}\"'
    )

    raw = call_gemini(prompt)
    parsed = safe_parse_json(raw)
    labels.append(parsed.get("label", "neutral").lower())
    scores.append(float(parsed.get("score", 0.0)))

    print(f"Processed {i+1}/{len(df)} → {parsed}")

    time.sleep(1)  # avoid rate limiting

# ---------- STEP 5: Save results ----------
df["predicted_sentiment"] = labels
df["sentiment_score"] = scores

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/news_sentiment_report.csv", index=False)
df.to_excel("outputs/news_sentiment_report.xlsx", index=False)

print("\n✅ Sentiment analysis complete!")
print("📂 Results saved to: outputs/news_sentiment_report.csv and .xlsx")