import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="News Sentiment Dashboard", layout="wide")

st.title("📰 News Sentiment Analysis Dashboard")
st.write("Module 4 Deployment — Streamlit App")

# -------------------------------
# Load Files
# -------------------------------

sentiment_file = "news_sentiment_report.csv"
forecast_file = "03_sentiment_forecast_7day.csv"
mock_file = "news_sentiment_report_7day_mock.csv"

# ==================================================
# SECTION 1 — LATEST SENTIMENT REPORT
# ==================================================
st.header("📌 Latest News Sentiment Report")

if os.path.exists(sentiment_file):
    df_sent = pd.read_csv(sentiment_file)

    # Convert timestamp
    df_sent["publishedAt"] = pd.to_datetime(df_sent["publishedAt"], errors="coerce")

    st.subheader("Raw Sentiment Table")
    st.dataframe(df_sent)

    # Line chart of sentiment score
    if "sentiment_score" in df_sent.columns:
        fig = px.line(
            df_sent.sort_values("publishedAt"),
            x="publishedAt",
            y="sentiment_score",
            title="Daily Sentiment Score",
        )
        st.plotly_chart(fig)
    else:
        st.error("Column 'sentiment_score' not found.")

else:
    st.warning(f"File not found: {sentiment_file}")


# ==================================================
# SECTION 2 — 7-DAY MOCK SENTIMENT DATA
# ==================================================
st.header("📌 7-Day Mock Data (Module 3 Input)")

if os.path.exists(mock_file):
    df_mock = pd.read_csv(mock_file)
    df_mock["publishedAt"] = pd.to_datetime(df_mock["publishedAt"], errors="coerce")

    st.dataframe(df_mock)

    if "sentiment_score" in df_mock.columns:
        fig2 = px.bar(
            df_mock.sort_values("publishedAt"),
            x="publishedAt",
            y="sentiment_score",
            title="7-Day Sentiment Score (Mock Data)"
        )
        st.plotly_chart(fig2)
else:
    st.warning(f"File not found: {mock_file}")


# ==================================================
# SECTION 3 — FORECAST RESULTS (PROPHET)
# ==================================================
st.header("📈 Sentiment Forecast Results (30-Day Prediction)")

if os.path.exists(forecast_file):
    df_forecast = pd.read_csv(forecast_file)

    # Prophet output always has 'ds' and 'yhat' columns
    if "ds" in df_forecast.columns:
        df_forecast["ds"] = pd.to_datetime(df_forecast["ds"], errors="coerce")

        st.subheader("Forecast Table")
        st.dataframe(df_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

        fig3 = px.line(
            df_forecast,
            x="ds",
            y=["yhat", "yhat_lower", "yhat_upper"],
            title="30-Day Sentiment Forecast"
        )
        st.plotly_chart(fig3)
    else:
        st.error("Forecast file does not contain 'ds' column!")

else:
    st.warning(f"File not found: {forecast_file}")


# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.write("✅ Streamlit deployment complete — Module 4 finished.")
