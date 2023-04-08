import string
from re import findall

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from config import token
from data.constants import PROJECT_PATH
from neural_network.neural_test import predict_spam


badwords = set()
with open(f"{PROJECT_PATH}/tg_bot/cenz.json", "r", encoding="utf-8") as cenz:
   badwords = set(eval(cenz.read()))

bot = Bot(token=token)
dp = Dispatcher(bot)


@dp.message_handler(commands = ["start", "help"])
async def cmd_start(message: types.Message):
    await message.answer("Бот работает")


@dp.message_handler(commands=["AddWord"])
async def add_cenz_filter(message: types.Message):
    badwords.add(message.text.lower()[9:].translate(str.maketrans("", "", string.punctuation)))

    with open(f"{PROJECT_PATH}/tg_bot/cenz.json", "w", encoding="utf-8") as jn:
        jn.write(str(badwords))
    await message.reply(f'Слово "{message.text[9:]}" добавлено в список')


@dp.message_handler()
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
