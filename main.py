import asyncio
import logging
import time

from aiogram import types, Bot, Dispatcher
from aiogram.filters.command import Command

import openai_core
import sql_worker
import utils

config = utils.ConfigData()
bot = Bot(token=config.token)
dp = Dispatcher()
sql_helper = sql_worker.SqlWorker(config.path + "database.db")

dialogs = {}
flood_wait = {}


def is_flooded(message):
    last_msg_time = flood_wait.get(message.from_user.id)
    if last_msg_time is None:
        flood_wait.update({message.from_user.id: int(time.time())})
        return False
    if int(time.time()) - last_msg_time <= 5:
        logging.info(f"Rejected request from user {utils.username_parser(message)} "
                     f"by floodwait. Retry after 5 seconds.")
        return True
    flood_wait.update({message.from_user.id: int(time.time())})
    return False


@dp.message(Command("start"))
async def start(message: types.Message):
    if not await utils.check_whitelist(message, config):
        return
    await message.reply("Я не бот, я живой человек! Не забывай об этом во время разговора со мной!")


@dp.message(lambda message: utils.check_names(message, config.my_id, config.prompts))
async def chatgpt(message: types.Message):
    if not await utils.check_whitelist(message, config):
        return
    context = message.chat.id if not config.unified_context else 0
    if dialogs.get(context) is None:
        dialogs.update({context: openai_core.Dialog(config, sql_helper, context)})
    if is_flooded(message):
        return
    logging.info(f"User {utils.username_parser(message)} send a request to ChatGPT")
    await bot.send_chat_action(chat_id=message.chat.id, action='typing')
    await dialogs.get(context).get_answer(message)


async def main() -> None:
    config.my_id = (await bot.get_me()).id
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.info("###HUMANOTRONIC v1.5 LAUNCHED SUCCESSFULLY###")
    asyncio.run(main())
