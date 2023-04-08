import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import numpy as np
import re
import constants

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


# Example usage:
forecast = predict_spam('Поздравляем, Вы выиграли бесплатную поездку на Гавайи!')
print(forecast)  # Output: Спам с вероятностью 98.51%

forecast = predict_spam('Привет, как ты поживаешь? Давно не виделись!')
print(forecast)  # Output: Не спам с вероятностью 4.99%

forecast = predict_spam('Вы выиграли 3 миллиона долларов! Ответьте на это письмо, чтобы получить свой приз')
print(forecast)  # Output: Спам с вероятностью 97.8%

forecast = predict_spam('ПОБЕДИТЕЛЬ!! Как ценный клиент сети, вы были выбраны для получения приза в размере 900 фунтов'
                        ' стерлингов! Чтобы подать заявку, позвоните по номеру 09061701461. '
                        'Код заявки KL341. Действует только 12 часов.')
print(forecast)  # Output: спам с вероятностью 0.34%

forecast = predict_spam("You won three million dollars in cash, do you want to take it back!")
print(forecast)  # Output: Spam

forecast = predict_spam("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to "
                        "receive entry question(std txt rate)T&C's apply 08452810075over18's")
print(forecast)  # Output: Spam

forecast = predict_spam("Congratulations, you have won a free trip to Hawaii!")
print(forecast)  # Output: Spam

forecast = predict_spam("Hi John, how are you doing today?")
print(forecast)  # Output: Ham