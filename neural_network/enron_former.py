import os
import csv
import asyncio
import aiofiles

from OrdoHereticus_bot import constants


dir_path = f'{constants.PROJECT_PATH}/dataset/enron/ham'

for ham in os.listdir(dir_path)[:5]:
    print(ham)
    with open(f"{dir_path}/{ham}", 'r') as ham_file:

        text = ham_file.read()
        with open(f'{constants.PROJECT_PATH}/dataset/enron.csv', 'a', encoding='cp1251') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([0, text])
