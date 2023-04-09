import string
from re import findall
from pathlib import Path

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from tg_bot.config import token
from neural_network.neural_test import predict_spam


# путь до папки с проектом "OrdoHereticus_bot"
PROJECT_PATH = Path(__file__).resolve().parent.parent

badwords = set()
with open(f"{PROJECT_PATH}/tg_bot/cenz.json", "r", encoding="utf-8") as cenz:
   badwords = set(eval(cenz.read()))

bot = Bot(token=token)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# Функция для защиты от флуда
async def anti_flood(message: types.Message, *args, **kwargs):
    await message.delete()


@dp.message_handler(commands = ["start", "help"])
@dp.throttled(anti_flood, rate=2)
async def cmd_start(message: types.Message):
    await message.answer("Бот работает")


@dp.message_handler(commands=["AddWord"])
@dp.throttled(anti_flood, rate=2)
async def add_cenz_filter(message: types.Message):
    badwords.add(message.text.lower()[9:].translate(str.maketrans("", "", string.punctuation)))

    with open(f"{PROJECT_PATH}/tg_bot/cenz.json", "w", encoding="utf-8") as jn:
        jn.write(str(badwords))
    await message.reply(f'Слово "{message.text[9:]}" добавлено в список')


@dp.message_handler()
@dp.throttled(anti_flood, rate=2)
async def cenz_filter(message: types.Message):
    need_ai = True

    for dirty_word in message.text.lower().split(" "):
        clear_word = dirty_word.translate(str.maketrans("", "", string.punctuation))
        if clear_word in badwords:
            await message.reply("Выражайся культурно!!!" )
            await message.delete()

            need_ai = False
            break

        if findall('[a-z]', clear_word) and findall('[а-я]', clear_word):
            await message.reply("Выражайся культурно!!!")
            await message.delete()

            need_ai = False
            break

    if need_ai:
        if predict_spam(message.text):
            await message.reply('Не спамь!!!')
            await message.delete()


if __name__ == '__main__':
    # executor.start_polling(dp, skip_updates=True)
    executor.start_polling(dp)
