from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")


from transformers import pipeline

def get_sentiment_label(query):
    """
    Perform sentiment analysis on the input text and return the sentiment label.

    Parameters:
    - query (str): The text to be analyzed for sentiment.

    Returns:
    - str: The sentiment label of the input text ('POSITIVE', 'NEGATIVE', or 'NEUTRAL').
    """
    # Path to the sentiment analysis model
    model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    
    # Load the sentiment analysis model and tokenizer
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
    
    # Perform sentiment analysis on the input query
    res = sentiment_task(query)
    
    # Return the sentiment label of the analyzed text
    return res[0]['label']


# Example usage:
# query = "Covid cases are increasing fast!"
# sentiment_label = get_sentiment_label(query)
# print("Sentiment Label:", sentiment_label)
