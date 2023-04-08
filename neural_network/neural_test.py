import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Embedding, Dropout


# Preprocess message depending on the language
def preprocess_message(text):
    # If the message is in English
    if all(re.match('[a-z]', c) for c in text):
        text = re.sub('<[^>]*>', '', text)  # Remove HTML tags if any
        text = re.sub('[^a-zA-z0-9\s]', '', text)  # Keep only alphanumeric characters and spaces
        text = text.lower()  # Convert to lower case
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('english')]  # Remove stop words
        return ' '.join(words)
    # If the message is in Russian
    else:
        text = re.sub('<[^>]*>', '', text)  # Remove HTML tags if any
        text = re.sub('[^а-яё0-9\s]', '', text)  # Keep only alphanumeric characters and spaces
        text = text.lower()  # Convert to lower case
        words = text.split()
        words = [word for word in words if word not in stopwords.words('russian')]  # Remove stop words
        return ' '.join(words)


mails = pd.read_csv('../dataset/new_spam.csv', encoding='cp1251', on_bad_lines='skip')


max_features = 3000  # Top most words that will be considered
tokenizer = Tokenizer(num_words=max_features, split=' ')
max_len = 1000  # Defining the maximum length of sentence


# To classify new messages in the future, load the trained model
loaded_model = tf.keras.models.load_model('spam_classifier.h5')


# Then preprocess the message and classify it
def predict_spam(notification):
    notification = preprocess_message(notification)
    sequence = tokenizer.texts_to_sequences([notification])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = loaded_model.predict(padded)
    return 'Spam' if prediction[0][0] >= 0.5 else 'Ham'


# Example usage:
forecast = predict_spam("Поздравляем, вы выиграли бесплатную поездку на Гавайи!")
print(forecast)  # Output: Spam

forecast = predict_spam("Привет, как ты поживаешь?")
print(forecast)  # Output: Ham

forecast = predict_spam("You won three million dollars in cash, do you want to take it back!")
print(forecast)  # Output: Spam

forecast = predict_spam("Какая сегодня погода?")
print(forecast)  # Output: Ham