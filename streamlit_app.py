import streamlit as st
from PIL import Image

st.title("AI News Sentiment Forecast")
img = Image.open("outputs/forecast.png")
st.image(img, caption="Forecast with 95% uncertainty", use_column_width=True)
