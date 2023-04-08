from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from constants import PROJECT_PATH
from config import token
from neural_network.neural_test import predict_spam
import string
from re import findall


badwords = set()
with open(f"{PROJECT_PATH}/tg_bot/cenz.json", "r", encoding="utf-8") as cenz:
   badwords = set(eval(cenz.read()))

bot = Bot(token=token)
dp = Dispatcher(bot)


@dp.message_handler(commands = ["start", "help"])
async def cmd_start(message: types.Message):
    await message.answer("Бот работает")


@dp.message_handler(commands=["AddWord"])
async def Add_Сenz_Filter(message: types.Message):
    badwords.add(message.text.lower()[9:]\
    .translate(str.maketrans("", "", string.punctuation)))
    with open(f"{PROJECT_PATH}/tg_bot/cenz.json", "w", encoding="utf-8") as jn:
        jn.write(str(badwords))
    await message.reply("Слово " + message.text[9:] + " добавлено в список")


@dp.message_handler()
async def Сenz_Filter(message: types.Message):
    for i in message.text.lower().split(" "):
        word = i.translate(str.maketrans("", "", string.punctuation))

        if word in badwords:
            await message.reply("ЗА ИМПЕРАТОРА!!!" )
            await message.delete()
        if findall('[a-z]', word) and findall('[а-я]', word):
            await message.reply("ЗА ИМПЕРАТОРА!!!")
            await message.delete()


@dp.message_handler()
async def spam_filter(message: types.Message):
    if predict_spam(message):
        await message.reply('Не спамь!')
        await message.delete()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
