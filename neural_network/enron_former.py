import os
import csv
import asyncio
import aiofiles

from googletrans import Translator
from OrdoHereticus_bot import constants


dir_path = f'{constants.PROJECT_PATH}/dataset/enron/ham'
translator = Translator()

with open(f'{constants.PROJECT_PATH}/dataset/enron.csv', 'a', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['labels', 'message'])

for ham in os.listdir(dir_path):
    print(ham)
    with open(f"{dir_path}/{ham}", 'r') as ham_file:

        file_text = ham_file.read()
        with open(f'{constants.PROJECT_PATH}/dataset/enron.csv', 'a', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([0, file_text])
            writer.writerow([0, translator.translate(file_text, src='en', dest='ru').text])
