import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.utils import pad_sequences
import constants
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Embedding


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
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('russian')]  # Remove stop words
        return ' '.join(words)


mails = pd.read_csv(f'{constants.PROJECT_PATH}/dataset/new_spam.csv', encoding='cp1251', on_bad_lines='skip')

message = mails['message'].tolist()
label = mails['label'].tolist()

# Preprocess the messages
message = [preprocess_message(text) for text in message]

# Tokenizing the messages
max_features = 3000  # Top most words that will be considered
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(message)
seq = tokenizer.texts_to_sequences(message)

# Padding to approach all messages to the same length
max_len = 1000  # Defining the maximum length of sentence
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

# Define a lambda function to calculate accuracy on the test set after each epoch
calc_test_accuracy = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f' - Test accuracy: {logs["val_accuracy"]*100}%\n'))

# Train the model with the callback to calculate accuracy at the end of each epoch
model.fit(X_train, y_train, epochs=5, batch_size=64,
          validation_split=0.2, callbacks=[calc_test_accuracy])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy*100}%')

# Save the model for later use
model.save('spam_classifier.h5')

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
