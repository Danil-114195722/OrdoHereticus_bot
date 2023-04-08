from langdetect.lang_detect_exception import LangDetectException
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import tensorflow as tf
import pandas as pd
import langdetect
import constants
import re


# Create labels language
labels_en = ['не спам', 'спам']
labels_ru = ['спам', 'не спам']
# Read the dataset from CSV
data = pd.read_csv(f'{constants.PROJECT_PATH}/dataset/spam_russian.csv')

# Extract columns for message and label
message_cols = ['message', 'message_2']
label_col = 'label'

messages = [data[column].tolist() for column in message_cols]
label = [int(i == 'spam') for i in data[label_col]]

# Tokenizing the messages
max_features = 2000  # Top most words that will be considered
tokenizer = Tokenizer(num_words=max_features, split=' ')
# concatenate the messages from tuples and store in a list
messages = [" ".join(msg_tuple) for msg_tuple in zip(*messages)]
# tokenize the messages
tokenizer.fit_on_texts(messages)
seq = tokenizer.texts_to_sequences(messages)

# Padding to approach all messages to the same length
max_len = 100  # Defining the maximum length of sentence
X = pad_sequences(seq, maxlen=max_len)

# To classify new messages in the future, load the trained model
loaded_model = tf.keras.models.load_model(f'{constants.PROJECT_PATH}/neural_network/spam_classifier.h5')


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
        return f'{labels_ru[int(round(prediction[0][0]))]}'
    else:
        return f'{labels_en[int(round(prediction[0][0]))]}'
