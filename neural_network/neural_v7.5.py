import tensorflow as tf
from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import numpy as np
import re
import constants

# Create labels language
labels_en = [False, True]
labels_ru = [True, False]
# Read the dataset from CSV
data_spam_russian = pd.read_csv(f'{constants.PROJECT_PATH}/dataset/spam_russian.csv')
data_enron = pd.read_csv(f'{constants.PROJECT_PATH}/dataset/shuffle_enron.csv')

# Extract columns for message and label
enron_messages = data_enron['message'].tolist()
message_cols = ['message', 'message_2']
label_col = 'label'
# We clear the text of messages from HTML tags using regular expressions
for i in range(len(enron_messages)):
    enron_messages[i] = re.sub('<[^>]*>', '', enron_messages[i])
    enron_messages[i] = re.sub('[^a-zA-Z0-9\s]+', '', enron_messages[i])

messages = [data_spam_russian[column].tolist() for column in message_cols]
label = [int(i) for i in data_spam_russian[label_col]]

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

tokenizer_en = Tokenizer(num_words=max_features, split=' ')
tokenizer_en.fit_on_texts(enron_messages)
seq_enron = tokenizer_en.texts_to_sequences(enron_messages)
X_enron = pad_sequences(seq_enron, maxlen=max_len)

# We glue messages and tags from two datasets
messages = messages + enron_messages
labels = [int(j) for j in data_enron['labels']]
label += labels

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
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
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
