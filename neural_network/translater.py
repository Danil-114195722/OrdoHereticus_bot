import pandas as pd
from googletrans import Translator
from OrdoHereticus_bot import constants

import asyncio


def translate_csv(from_file: str, to_file: str) -> None:
    df = pd.read_csv(from_file, encoding='cp1251')
    translator = Translator()

    russian_texts = []
    for num, texts in enumerate(df['message'], 0):
        russian_text = translator.translate(texts, src='en', dest='ru', encodings='cp1251').text
        russian_texts.append(russian_text)
        print(russian_text)
        print(num)
    df['russian_text'] = russian_texts
    df.to_csv(to_file, index=False)


if __name__ == '__main__':
    translate_csv(
        f'{constants.PROJECT_PATH}dataset/spam.csv',
        f'{constants.PROJECT_PATH}dataset/spam_russian.csv'
    )
