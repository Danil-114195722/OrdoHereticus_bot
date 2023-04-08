import pandas as pd
from googletrans import Translator

df = pd.read_csv('./spam.csv')

translator = Translator()

russian_texts = []
for num, texts in enumerate(df['message'],0):
    russian_text = translator.translate(texts, src='en', dest='ru').text
    russian_texts.append(russian_text)
    print(russian_text)
    print(num)
df['russian_text'] = russian_texts
df.to_csv('spam_russian.csv', index=False)