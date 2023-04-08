import os
import csv
import asyncio
import aiofiles

from httpcore._exceptions import ReadTimeout
from googletrans import Translator
import constants


DIR_PATH = f'{constants.PROJECT_PATH}/dataset/enron/spam'
TRANSLATOR = Translator()


async def add_en_ru(spam):
    global errors, insoluble_errors

    async with aiofiles.open(f"{DIR_PATH}/{spam}", 'r') as spam_file:
        file_text = await spam_file.read()

        async with aiofiles.open(f'{constants.PROJECT_PATH}/dataset/enron.csv', 'a', encoding='utf-8') as csv_file:
            try:
                attempts = 0
                while attempts <= 5:
                    try:
                        writer = csv.writer(csv_file)
                        await writer.writerow([0, file_text])
                        print(spam)
                        await writer.writerow([0, TRANSLATOR.translate(file_text, src='en', dest='ru').text])
                        break
                    except ValueError:
                        errors += 1
                        if attempts == 5:
                            insoluble_errors += 1
            except ReadTimeout:
                insoluble_errors += 1


async def main():
    # Создание заголовка
    # async with aiofiles.open(f'{constants.PROJECT_PATH}/dataset/enron.csv', 'a', encoding='utf-8') as csv_file:
    #     writer = csv.writer(csv_file)
    #     await writer.writerow(['labels', 'message'])

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
