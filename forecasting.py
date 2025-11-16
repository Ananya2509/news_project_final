import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# *** CHANGE THIS LINE ***
INPUT_FILE = 'news_sentiment_report_7day_mock.csv' 
OUTPUT_FILE = '03_sentiment_forecast_7day.csv'
FIGURE_FILE = '03_sentiment_forecast_plot_7day.png'

def forecast_sentiment():
    """Prepares data, fits the Prophet model, and forecasts sentiment."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found. Ensure file is in the project folder.")
        return

    # 1. Load and Prepare Data
    df = pd.read_csv(INPUT_FILE)
    
    # *** USING 'publishedAt' and 'sentiment_score' from your file ***
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # Aggregate daily average score using 'publishedAt'
    daily_sentiment = df.groupby(df['publishedAt'].dt.date)['sentiment_score'].mean().reset_index()
    
    # Rename columns for Prophet (ds = date, y = value to forecast)
    daily_sentiment.columns = ['ds', 'y'] 
    daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
    
    print(f"Aggregated daily data for {len(daily_sentiment)} days.")

    # 2. Initialize and Fit Prophet Model
    model = Prophet(
        yearly_seasonality=False, 
        weekly_seasonality=True, 
        daily_seasonality=False, 
        growth='linear' 
    ) 
    model.fit(daily_sentiment)
    
    # 3. Create Future DataFrame and Predict
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # 4. Save and Plot Results
    forecast.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Sentiment forecast complete. Results saved to '{OUTPUT_FILE}'")

    # Plotting the forecast
    fig = model.plot(forecast)
    plt.title("Sentiment Score Forecast (with 30-day lookahead)")
    plt.savefig(FIGURE_FILE)
    print(f"✅ Forecast chart saved to '{FIGURE_FILE}'")
    plt.close()

if __name__ == "__main__":
    forecast_sentiment()