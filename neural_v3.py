import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re


mails = pd.read_csv('./spam.csv', encoding='latin-1')
message = mails['message'].tolist()
label = mails['label'].tolist()
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(message)
seq = tokenizer.texts_to_sequences(message)
max_len = 100
X = pad_sequences(seq, maxlen=max_len)
loaded_model = tf.keras.models.load_model('spam_classifier.h5')


def predict_spam(notification):
    notification = re.sub('<[^>]*>', '', notification)
    notification = re.sub('[^a-zA-z0-9\s]', '', notification)
    notification = notification.lower()
    sequence = tokenizer.texts_to_sequences([notification])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = loaded_model.predict(padded)
    return 'Spam' if prediction[0][0] >= 0.8 else 'Ham'


forecast = predict_spam("Поздравляем, вы выиграли бесплатную поездку на Гавайи!")
print(forecast)

forecast = predict_spam("Привет Джон, как дела сегодня?")
print(forecast)
