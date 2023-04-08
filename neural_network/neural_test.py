import re
import pandas as pd

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

import langdetect
from langdetect.lang_detect_exception import LangDetectException

from data.constants import PROJECT_PATH


# Create labels language
labels_en = [False, True]
labels_ru = [True, False]
# Read the dataset from CSV
data_spam_russian = pd.read_csv(f'{PROJECT_PATH}/dataset/spam_russian.csv')
data_enron = pd.read_csv(f'{PROJECT_PATH}/dataset/enron.csv')

# Extract columns for message and label
enron_messages = data_enron['message'].tolist()
message_cols = ['message', 'message_2']
label_col = 'label'
# We clear the text of messages from HTML tags using regular expressions
for i in range(len(enron_messages)):
    enron_messages[i] = re.sub('<[^>]*>', '', enron_messages[i])
    enron_messages[i] = re.sub('[^a-zA-Z0-9\s]+', '', enron_messages[i])

messages = [data_spam_russian[column].tolist() for column in message_cols]
label = [int(i == 'spam') for i in data_spam_russian[label_col]]

# Tokenizing the messages
max_features = 2000  # Top most words that will be considered
tokenizer = Tokenizer(num_words=max_features, split=' ')
# concatenate the messages from tuples and store in a list
messages = [" ".join(msg_tuple) for msg_tuple in zip(*messages)]
# tokenize the messages
tokenizer.fit_on_texts(messages)
seq = tokenizer.texts_to_sequences(messages)

# Padding to approach all messages to the same length
max_len = 500  # Defining the maximum length of sentence
X = pad_sequences(seq, maxlen=max_len)

# To classify new messages in the future, load the trained model
loaded_model = tf.keras.models.load_model(f'{PROJECT_PATH}/neural_network/spam_classifier.h5')


def predict_spam(message):
    if isinstance(message, list):
        message = " ".join(message)
    message = re.sub('<[^>]*>', '', message)  # Remove HTML tags if any
    message = re.sub('[^a-zа-яёA-ZА-Я0-9\s]', '', message)  # Keep only alphanumeric characters and spaces
    message = message.lower()  # Convert to lower case
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = loaded_model.predict(padded)
    try:
        lang = langdetect.detect(message)
    except LangDetectException:
        lang = ''
    if lang == 'ru':
        return labels_ru[int(round(prediction[0][0]))]
    else:
        return labels_en[int(round(prediction[0][0]))]
