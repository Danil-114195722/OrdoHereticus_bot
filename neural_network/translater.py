import pandas as pd
from googletrans import Translator
from tg_bot import constants


df = pd.read_csv(f'{constants.PROJECT_PATH}/dataset/spam.csv', encoding='cp1251')
translator = Translator()

russian_texts = []
for num, texts in enumerate(df['message'], 0):
    russian_text = translator.translate(texts, src='en', dest='ru', encodings='cp1251').text
    russian_texts.append(russian_text)
    print(russian_text)
    print(num)
df['russian_text'] = russian_texts
df.to_csv(f'{constants.PROJECT_PATH}/dataset/spam_russian.csv', index=False)
