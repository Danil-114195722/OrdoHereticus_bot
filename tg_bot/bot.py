from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from config import token
import constants

badwords = []
with open(f"{constants.PROJECT_PATH}/tg_bot/cenz.json", "r", encoding="utf-8") as cenz:
   badwords = eval(cenz.read())

bot = Bot(token=token)
dp = Dispatcher(bot)


@dp.message_handler(commands = ["start", "help"])
async def cmd_start(message: types.Message):
    await message.answer("Бот работает")


@dp.message_handler(commands=["AddWord"])
async def Add_Сenz_Filter(message: types.Message):
    badwords.append(message.text.lower()[9:])
    with open("cenz.json", "w", encoding="utf-8") as jn:
        jn.write(str(badwords))
    await message.reply("Слово " + message.text[9:] + " добавлено в список")


@dp.message_handler()
async def Сenz_Filter(message: types.Message):
    for i in message.text.lower().split(" "):
        if i in badwords:
            await message.reply("ЗА ИМПЕРАТОРА!!!")
            await message.delete()


if __name__ == '__main__':
    executor.start_polling(dp)