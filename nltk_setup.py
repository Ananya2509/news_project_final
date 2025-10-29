import nltk

# Download stopwords
nltk.download("stopwords")

# Example: use stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
print(stop_words)
