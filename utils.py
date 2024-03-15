import base64
import configparser
import importlib
import logging
import os
import re
import sys
import time
import traceback
from importlib import reload
from io import BytesIO
from PIL import Image

from aiogram import types


class ConfigData:
    path = ""
    my_id = ""

    def __init__(self):

        try:
            self.path = sys.argv[1] + "/"
            if not os.path.isdir(sys.argv[1]):
                print("WARNING: working path IS NOT EXIST. Remake.")
                os.mkdir(sys.argv[1])
        except IndexError:
            pass
        except IOError:
            traceback.print_exc()
            print("ERROR: Failed to create working directory! Bot will be closed!")
            sys.exit(1)

        reload(logging)
        logging.basicConfig(
            handlers=[
                logging.FileHandler(self.path + "logging.log", 'w', 'utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt="%d-%m-%Y %H:%M:%S")

        prompts_path = f"{self.path[:-1]}.prompts" if self.path else "prompts"
        if not os.path.isfile(f"{self.path}prompts.py") and self.path:
            logging.warning(f"The prompts file was not found in the {self.path}, a generic prompt will be used!")
            prompts_path = "prompts"
        try:
            self.prompts = importlib.import_module(prompts_path)
        except Exception as e:
            logging.error(f'Module prompts.py is invalid! {e}')
            logging.error(traceback.format_exc())
            sys.exit(1)

        if not os.path.isfile(self.path + "config.ini"):
            print("Config file isn't found! Trying to remake!")
            self.remake_conf()

        config = configparser.ConfigParser()
        while True:
            try:
                config.read(self.path + "config.ini")
                self.token = config["ChatGPT"]["token"]
                self.api_key = config["ChatGPT"]["api-key"]
                self.model = config["ChatGPT"]["model"]
                self.whitelist = config["ChatGPT"]["whitelist-chats"]
                self.timezone = int(config["ChatGPT"]["timezone"])
                self.unified_context = self.bool_init(config["ChatGPT"]["unified-context"])
                self.summarizer_limit = int(config["ChatGPT"]["summarizer-limit"])
                self.tokens_per_answer = int(config["ChatGPT"]["tokens-per-answer"])
                self.memory_dump_size = int(config["ChatGPT"]["memory-dump-size"])
                self.vision = self.bool_init(config["ChatGPT"]["vision"])
                self.stream_mode = self.bool_init(config["ChatGPT"]["stream-mode"])
                break
            except Exception as e:
                logging.error((str(e)))
                logging.error(traceback.format_exc())
                time.sleep(1)
                print("\nInvalid config file! Trying to remake!")
                agreement = "-1"
                while agreement != "y" and agreement != "n" and agreement != "":
                    agreement = input("Do you want to reset your broken config file on defaults? (Y/n): ")
                    agreement = agreement.lower()
                if agreement == "" or agreement == "y":
                    self.remake_conf()
                else:
                    sys.exit(0)
        try:
            self.base_url = config["ChatGPT"]["base-url"]
            if self.base_url == "":
                raise KeyError
        except (KeyError, TypeError):
            self.base_url = None

        try:
            self.temperature = float(config["ChatGPT"]["temperature"])
            if self.temperature == "":
                raise KeyError
        except (KeyError, TypeError, ValueError):
            self.temperature = None

    def remake_conf(self):
        token, api_key, model = "", "", ""
        while token == "":
            token = input("Please, write your bot token: ")
        while api_key == "":
            api_key = input("Please, write your OpenAI developer key: ")
        while model == "":
            model = input("Please, write your ChatGPT model name: ")
        config = configparser.ConfigParser()
        config.add_section("ChatGPT")
        config.set("ChatGPT", "token", token)
        config.set("ChatGPT", "whitelist-chats", "")
        config.set("ChatGPT", "unified-context", "false")
        config.set("ChatGPT", "api-key", api_key)
        config.set("ChatGPT", "base-url", "")
        config.set("ChatGPT", "model", model)
        config.set("ChatGPT", "temperature", "0.5")
        config.set("ChatGPT", "timezone", "0")
        config.set("ChatGPT", "stream-mode", "false")
        if "gpt-4" in model or "16k" in model or "32k" in model:
            config.set("ChatGPT", "summarizer-limit", "6000")
            config.set("ChatGPT", "tokens-per-answer", "2000")
            config.set("ChatGPT", "memory-dump-size", "2000")
        else:
            config.set("ChatGPT", "summarizer-limit", "2500")
            config.set("ChatGPT", "tokens-per-answer", "1000")
            config.set("ChatGPT", "memory-dump-size", "1000")
        if "vision" in model:
            config.set("ChatGPT", "vision", "true")
        else:
            config.set("ChatGPT", "vision", "false")
        try:
            config.write(open(self.path + "config.ini", "w"))
            print("New config file was created successful")
        except IOError:
            print("ERR: Bot cannot write new config file and will close")
            logging.error(traceback.format_exc())
            sys.exit(1)

    @staticmethod
    def bool_init(var):
        if var.lower() in ("false", "0"):
            return False
        elif var.lower() in ("true", "1"):
            return True
        else:
            raise TypeError


def check_names(message, bot_id, prompts):
    """
    The bot will only respond if called by name (if it's public chat)
    :param message:
    :param bot_id:
    :param prompts:
    :return:
    """

    if all([message.text is None, message.photo is None, message.sticker is None]):
        return False
    if message.chat.id == message.from_user.id:
        return True
    if message.reply_to_message:
        if message.reply_to_message.from_user.id == bot_id:
            return True
    msg_txt = message.text or message.caption
    if msg_txt is None:
        return False
    msg_txt = re.sub(r'[^\w\s]', '', msg_txt.lower()).split()
    for name in prompts.names:
        if name.lower() in msg_txt:
            return True
    return False


def username_parser(message, html=False):
    if message.from_user.first_name == "":
        return "DELETED USER"

    if message.from_user.username == "GroupAnonymousBot":
        return "ANONYMOUS ADMIN"

    if message.from_user.last_name is None:
        username = str(message.from_user.first_name)
    else:
        username = str(message.from_user.first_name) + " " + str(message.from_user.last_name)

    if not html:
        return username

    return html_fix(username)


def html_fix(text):
    """
    Fixes some characters that could cause problems with parse_mode=html
    :param text:
    :return:
    """
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def current_time_info(config, int_time: int = None):
    if not int_time:
        int_time = int(time.time())
    current_time = time.strftime("%d.%m.%Y, %a, %H:%M", time.gmtime(int_time + config.timezone * 3600))
    return f"Current time: {current_time}"


async def check_whitelist(message: types.Message, config):
    if str(message.chat.id) in config.whitelist or not config.whitelist:
        return "True"
    private = "с тобой" if message.chat is None else "здесь"
    # noinspection PyUnresolvedReferences
    chat_name = username_parser(message) if message.chat.title is None else message.chat.title
    logging.info(f"Rejected request from chat {chat_name}")
    await message.reply(f"Извини, но мне нельзя говорить {private}. Это не моя вина, просто на всех не разорваться.")
    return False


def get_image_width(photo_base64):
    width, _ = Image.open(BytesIO(base64.b64decode(photo_base64.split("base64,")[-1]))).size
    return width
