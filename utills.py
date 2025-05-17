import csv
import os

def format_sentiment(sentiment):
    """
    Format sentiment label for HTML display.

    Parameters:
    - sentiment (str): Sentiment label ('positive', 'neutral', or 'negative').

    Returns:
    - str: HTML-formatted sentiment label.
    """
    if sentiment == 'positive':
        return '<span style="color:green; font-weight:bold;">Positive</span>'
    elif sentiment == 'neutral':
        return '<span style="color:yellow; font-weight:bold;">Neutral</span>'
    elif sentiment == 'negative':
        return '<span style="color:red; font-weight:bold;">Negative</span>'
    else:
        return 'Invalid sentiment'


def save_to_csv(Question, Response, Sentiment, filename='output.csv'):
    """
    Save question, response, and sentiment to a CSV file.

    Parameters:
    - Question (str): The question text.
    - Response (str): The response text.
    - Sentiment (str): The sentiment label ('positive', 'neutral', or 'negative').
    - filename (str): The name of the CSV file to save to. Defaults to 'output.csv'.
    """
    # Check if the file exists, if not create it and write the headers
    is_new_file = not os.path.exists(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(['Question', 'Response', 'Sentiment'])
        writer.writerow([Question, Response, Sentiment])



