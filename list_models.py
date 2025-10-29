import google.generativeai as genai

# Configure API credentials
genai.configure(api_key="AIzaSyDBg2UBLI5DLVPNL9uQxvqlVyNp_c0Isj0")  # Replace YOUR_API_KEY with the actual key

# List all models
models = genai.list_models()
for m in models:
    print(m.name, m.supported_generation_methods)
