import tensorflow as tf
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords


# Preprocess message depending on the language
def preprocess_message(text):
    # If the message is in English
    if all(ord(c) < 128 for c in text):
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


mails = pd.read_csv('./spam.csv', encoding='latin-1', error_bad_lines=False)

message = mails['message'].tolist()
label = mails['label'].tolist()

# Preprocess the messages
message = [preprocess_message(text) for text in message]

# Tokenizing the messages
max_features = 2000  # Top most words that will be considered
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(message)
seq = tokenizer.texts_to_sequences(message)

# Padding to approach all messages to the same length
max_len = 100  # Defining the maximum length of sentence
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
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Save the model for later use
model.save('spam_classifier.h5')

# To classify new messages in the future, load the trained model
loaded_model = tf.keras.models.load_model('spam_classifier.h5')


# Then preprocess the message and classify it
def predict_spam(message):
    message = preprocess_message(message)
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = loaded_model.predict(padded)
    return 'Spam' if prediction[0][0] >= 0.9 else 'Ham'


# Example usage:
prediction = predict_spam("Поздравляем, вы выиграли бесплатную поездку на Гавайи!")
print(prediction)  # Output: Spam

prediction = predict_spam("Привет, как ты поживаешь?")
print(prediction)  # Output: Ham