# from dotenv import load_dotenv
import json
import logging
import random
import threading
import time
import traceback

import anthropic

import utils


class Dialog:

    def __init__(self, config, sql_helper, context):
        self.dialogue_locker = False
        self.config = config
        self.sql_helper = sql_helper
        self.context = context
        try:
            dialog_data = sql_helper.dialog_get(context)
        except Exception as e:
            dialog_data = []
            logging.error("Humanotronic was unable to read conversation information! Please check your database!")
            logging.error(f"{e}\n{traceback.format_exc()}")
        if not dialog_data:
            start_time = f"{utils.current_time_info(config).split(maxsplit=2)[2]} - it's time to start our conversation"
            self.dialog_history = [{"role": "assistant", "content": "new"}]
        else:
            start_time = (f"{utils.current_time_info(config, dialog_data[0][2]).split(maxsplit=2)[2]} - "
                          f"it's time to start our conversation")
            self.dialog_history = json.loads(dialog_data[0][1])
            # Pictures saved in the database may cause problems when working without Vision
            if not config.vision:
                self.dialog_history = self.cleaning_images(self.dialog_history)
        self.system = f"{config.prompts.start}\n{config.prompts.hard}\n{start_time}"
        self.client = anthropic.Anthropic(api_key=config.api_key, base_url=config.base_url)

    def get_answer(self, message, reply_msg, photo_base64):
        chat_name = utils.username_parser(message) if message.chat.title is None else message.chat.title
        if reply_msg:
            if self.dialog_history[-1]['content'] == reply_msg['content']:
                reply_msg = None

        msg_txt = message.text or message.caption
        if msg_txt is None:
            msg_txt = "I sent a photo"

        prompt = ""
        if random.randint(1, 50) == 1:
            prompt += f"{self.config.prompts.prefill} "
            logging.info(f"Prompt reminded for dialogue in chat {chat_name}")
        if random.randint(1, 30) == 1 or "врем" in msg_txt.lower() or "час" in msg_txt.lower():
            prompt += f"{utils.current_time_info(self.config)} "
            logging.info(f"Time updated for dialogue in chat {chat_name}")
        prompt += f"{utils.username_parser(message)}: {msg_txt}"
        dialog_buffer = self.dialog_history.copy()[1::]
        if reply_msg:
            dialog_buffer.append(reply_msg)
        if photo_base64:
            dialog_buffer.append({"role": "user", "content": [
                {"type": "image", "source":
                    {"type": "base64", "media_type": photo_base64['mime'], "data": photo_base64['data']}},
                {"type": "text", "text": prompt}]})
        else:
            dialog_buffer.append({"role": "user", "content": prompt})
        summarizer_used = False
        while self.dialogue_locker is True:
            summarizer_used = True
            logging.info(f"Adding messages is blocked for chat {chat_name} "
                         f"due to the work of the summarizer. Retry after 5s.")
            time.sleep(5)
        try:
            completion = self.client.messages.create(
                model=self.config.model,
                messages=dialog_buffer,
                temperature=self.config.temperature,
                max_tokens=self.config.tokens_per_answer,
                stream=False,
                system=self.system
            )
            answer = completion.content[0].text
        except Exception as e:
            logging.error(f"{e}\n{traceback.format_exc()}")
            return random.choice(self.config.prompts.errors)

        total_tokens = completion.usage.input_tokens + completion.usage.output_tokens
        logging.info(f'{total_tokens} tokens counted by the Anthropic API in chat {chat_name}.')
        while self.dialogue_locker is True:
            summarizer_used = True
            logging.info(f"Adding messages is blocked for chat {chat_name} "
                         f"due to the work of the summarizer. Retry after 5s.")
            time.sleep(5)
        if reply_msg:
            self.dialog_history.append(reply_msg)
        if photo_base64:
            self.dialog_history.extend([{"role": "user", "content": [
                {"type": "image", "source":
                    {"type": "base64", "media_type": photo_base64['mime'], "data": photo_base64['data']}},
                {"type": "text", "text": prompt}]},
                                        {"role": "assistant", "content": str(answer)}])
        else:
            self.dialog_history.extend([{"role": "user", "content": prompt},
                                        {"role": "assistant", "content": str(answer)}])
        if self.config.vision and len(self.dialog_history) > 10:
            self.dialog_history = self.cleaning_images(self.dialog_history, last_only=True)
        if total_tokens >= self.config.summarizer_limit and not summarizer_used:
            logging.info(f"The token limit {self.config.summarizer_limit} for "
                         f"the {chat_name} chat has been exceeded. Using a lazy summarizer")
            threading.Thread(target=self.summarizer, args=(chat_name,)).start()
        try:
            self.sql_helper.dialog_update(self.context, json.dumps(self.dialog_history))
        except Exception as e:
            logging.error("Humanotronic was unable to save conversation information! Please check your database!")
            logging.error(f"{e}\n{traceback.format_exc()}")
        return answer

    # This code clears the context from old images so that they do not cause problems in operation
    # noinspection PyTypeChecker
    @staticmethod
    def cleaning_images(dialog, last_only=False):

        def cleaner():
            if isinstance(dialog[index]['content'], list):
                for i in dialog[index]['content']:
                    if i['type'] == 'text':
                        dialog[index]['content'] = i['text']

        if last_only:
            for index in range(len(dialog) - 11, -1, -1):
                cleaner()
        else:
            for index in range(len(dialog)):
                cleaner()
        return dialog

    def summarizer_index(self, dialogue, threshold=None):
        text_len = 0
        for index in range(len(dialogue)):
            if isinstance(dialogue[index]['content'], list):
                for i in dialogue[index]['content']:
                    if i['type'] == 'text':
                        text_len += len(i['text'])
            else:
                text_len += len(dialogue[index]['content'])

            if threshold:
                if text_len >= threshold and dialogue[index]['role'] == "user":
                    return index

        return self.summarizer_index(dialogue, text_len * 0.7)

    # noinspection PyTypeChecker
    def summarizer(self, chat_name):
        self.dialogue_locker = True
        if self.dialog_history[0]['content'] == 'brief':
            # The dialogue cannot begin with the words of the assistant, which means it was a diary entry
            last_diary = self.dialog_history[1]['content']
            dialogue = [{"role": "user", "content": "Dialogue is started"}]
            dialogue.extend(self.dialog_history[2::])
        else:
            last_diary = None
            dialogue = self.dialog_history[1::]

        summarizer_text = self.config.prompts.summarizer
        if last_diary is not None:
            summarizer_text += f"\n{self.config.prompts.summarizer_last}"
        summarizer_text = summarizer_text.format(self.config.memory_dump_size)

        split = self.summarizer_index(dialogue)

        compressed_dialogue = dialogue[:split:]
        if last_diary is None:
            compressed_dialogue.append({"role": "user", "content": f'{summarizer_text}\n{self.system}'
                                                                   f'\n{utils.current_time_info(self.config)}'})
        else:
            compressed_dialogue.append({"role": "user",
                                        "content": f'{summarizer_text}\n{last_diary}'
                                                   f'\n{utils.current_time_info(self.config)}'})

        # When sending pictures to the summarizer, it does not work correctly, so we delete them
        compressed_dialogue = self.cleaning_images(compressed_dialogue)
        original_dialogue = dialogue[split::]
        try:
            completion = self.client.messages.create(
                model=self.config.model,
                messages=compressed_dialogue,
                temperature=self.config.temperature,
                max_tokens=self.config.tokens_per_answer,
                stream=False,
            )
            answer = completion.content[0].text
        except Exception as e:
            logging.error(f"Summarizing failed for chat {chat_name}!")
            logging.error(f"{e}\n{traceback.format_exc()}")
            self.dialogue_locker = False
            return

        logging.info(f"Summarizing completed for chat {chat_name}, "
                     f"{completion.usage.input_tokens + completion.usage.output_tokens} tokens were used")
        result = [{"role": "assistant", "content": "brief"},
                  {"role": "user", "content": f'{answer}\n{utils.current_time_info(self.config)}'},
                  compressed_dialogue[-2]]
        result.extend(original_dialogue)
        self.dialog_history = result
        try:
            self.sql_helper.dialog_update(self.context, json.dumps(self.dialog_history))
        except Exception as e:
            logging.error("Humanotronic was unable to save conversation information! Please check your database!")
            logging.error(f"{e}\n{traceback.format_exc()}")
        self.dialogue_locker = False
