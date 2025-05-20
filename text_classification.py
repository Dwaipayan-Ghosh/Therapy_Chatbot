from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Load once and reuse
sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def get_sentiment_label(text):
    result = sentiment_task(text)
    return result[0]["label"]
