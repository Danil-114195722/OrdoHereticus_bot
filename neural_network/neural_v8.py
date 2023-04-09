import re
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

import langdetect
from langdetect.lang_detect_exception import LangDetectException


# путь до папки с проектом "OrdoHereticus_bot"
PROJECT_PATH = Path(__file__).resolve().parent.parent

# Create labels language
labels_en = [False, True]
labels_ru = [True, False]

data_spam_russian = pd.read_csv(f'{PROJECT_PATH}/dataset/enron.csv')

message = data_spam_russian['message'].tolist()
label = data_spam_russian['labels'].tolist()

# Tokenizing the messages
max_features = 2000  # Top most words that will be considered
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(message)
seq = tokenizer.texts_to_sequences(message)

# Padding to approach all messages to the same length
max_len = 250  # Defining the maximum length of sentence
X = pad_sequences(seq, maxlen=max_len)

# Splitting the dataset into train and test
test_ratio = 0.25
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = np.array(label)[indices]

train_size = int(X.shape[0] * 0.75)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the model
embedding_dim = 32  # Dimension of the token embedding
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Save the model for later use
model.save('spam_classifier.h5')

# To classify new messages in the future, load the trained model
loaded_model = tf.keras.models.load_model(f'{PROJECT_PATH}/neural_network/spam_classifier.h5')


def predict_spam(notification):
    if isinstance(notification, list):
        notification = " ".join(notification)
    notification = re.sub('<[^>]*>', '', notification)  # Remove HTML tags if any
    notification = re.sub('[^a-zа-яёA-ZА-Я0-9\s]', '', notification)  # Keep only alphanumeric characters and spaces
    notification = notification.lower()  # Convert to lower case
    sequence = tokenizer.texts_to_sequences([notification])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = loaded_model.predict(padded)
    try:
        lang = langdetect.detect(notification)
    except LangDetectException:
        lang = ''
    if lang == 'ru':
        return labels_ru[int(round(prediction[0][0]))]
    else:
        return labels_en[int(round(prediction[0][0]))]

