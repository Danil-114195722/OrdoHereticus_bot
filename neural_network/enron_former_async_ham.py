import os
import csv
from pathlib import Path

import asyncio
import aiofiles

from httpcore._exceptions import ReadTimeout
# from googletrans import Translator


# путь до папки с проектом "OrdoHereticus_bot"
PROJECT_PATH = Path(__file__).resolve().parent.parent

DIR_PATH = f'{PROJECT_PATH}/dataset/enron/ham'
# TRANSLATOR = Translator()


async def add_en_ru(ham):
    global errors, insoluble_errors

    async with aiofiles.open(f"{DIR_PATH}/{ham}", 'r') as ham_file:
        file_text = await ham_file.read()

        async with aiofiles.open(f'{PROJECT_PATH}/dataset/enron.csv', 'a', encoding='utf-8') as csv_file:
            try:
                attempts = 0
                while attempts <= 5:
                    try:
                        writer = csv.writer(csv_file)
                        await writer.writerow([0, file_text])
                        print(ham)
                        # await writer.writerow([0, TRANSLATOR.translate(file_text, src='en', dest='ru').text])
                        break
                    except ValueError:
                        errors += 1
                        if attempts == 5:
                            insoluble_errors += 1
            except ReadTimeout:
                insoluble_errors += 1


async def main():
    # Создание заголовка
    async with aiofiles.open(f'{PROJECT_PATH}/dataset/enron.csv', 'a', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        await writer.writerow(['label', 'message'])

    tasks = []
    for ham in os.listdir(DIR_PATH):
        task = asyncio.create_task(add_en_ru(ham))
        tasks.append(task)

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    errors = 0
    insoluble_errors = 0
    asyncio.run(main())
    print('Amount errors:', errors)
    print('Amount insoluble errors:', insoluble_errors)
