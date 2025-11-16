import pandas as pd
import requests
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
# *** UPDATED INPUT FILE ***
INPUT_FILE = 'news_sentiment_report.csv' 
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
CRITICAL_SCORE_THRESHOLD = -0.4 # Define what a "critical" score is
TIME_WINDOW_HOURS = 24 # Check sentiment over the last 24 hours

def send_slack_alert(message, webhook_url):
    """Sends a formatted message to a Slack channel via Webhook."""
    if not webhook_url:
        print("Error: SLACK_WEBHOOK_URL not set in .env file.")
        return

    slack_payload = {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": "🚨 CRITICAL SENTIMENT ALERT 🚨"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": message}},
            {"type": "divider"}
        ]
    }
    
    try:
        response = requests.post(webhook_url, json=slack_payload)
        response.raise_for_status()
        print(f"✅ Slack alert sent successfully. Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Slack alert: {e}")

def check_and_alert():
    """Checks the latest sentiment data and triggers a Slack alert if critical."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found. Ensure file is in the project folder.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # *** USING 'publishedAt' and 'sentiment_score' from your file ***
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # 1. Filter to the latest time window
    latest_time = df['publishedAt'].max()
    critical_start_time = latest_time - pd.Timedelta(hours=TIME_WINDOW_HOURS)
    
    recent_df = df[df['publishedAt'] >= critical_start_time]
    
    if recent_df.empty:
        print(f"No data found in the last {TIME_WINDOW_HOURS} hours.")
        return
        
    # 2. Calculate average score
    avg_score = recent_df['sentiment_score'].mean()
    
    print(f"Average sentiment score in the last {TIME_WINDOW_HOURS} hours: {avg_score:.2f}")

    # 3. Check against threshold and alert
    if avg_score <= CRITICAL_SCORE_THRESHOLD:
        
        most_negative_article = recent_df.loc[recent_df['sentiment_score'].idxmin()]
        
        alert_message = (
            f"*Average Sentiment Score:* `{avg_score:.2f}` (Threshold: `{CRITICAL_SCORE_THRESHOLD}`)\n"
            f"*Time Window:* Last {TIME_WINDOW_HOURS} hours (since {critical_start_time.strftime('%Y-%m-%d %H:%M')})\n\n"
            f"*Most Negative Headline (Score {most_negative_article['sentiment_score']:.2f}):*\n"
            f"> {most_negative_article['title']}"
        )
        
        send_slack_alert(alert_message, SLACK_WEBHOOK_URL)
    else:
        print("Sentiment score is above the critical threshold. No alert sent.")

if __name__ == "__main__":
    check_and_alert()