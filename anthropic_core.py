# from dotenv import load_dotenv
import json
import logging
import random
import threading
import time
import traceback

import anthropic

import utils


class ApiRequestException(Exception):
    pass


class Dialog:

    def __init__(self, config, sql_helper, context):
        self.dialogue_locker = False
        self.config = config
        self.sql_helper = sql_helper
        self.context = context
        self.last_time = 0
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

    def send_api_request(self, model, messages,
                         max_tokens=1000,
                         system=None,
                         temperature=None,
                         stream=False):

        msg = []
        if system:
            msg = [{'role': 'user', "content": "Dialogue is started"}, {'role': 'assistant', "content": system}]
        msg.extend(messages)
        messages = msg
        messages.append({"role": "assistant",
                         "content": self.config.prompts.prefill})
        if not stream:
            try:
                completion = self.client.messages.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                if "error" in completion.id:
                    logging.error(completion.content[0].text)
                    raise ApiRequestException
                return completion.content[0].text, completion.usage.input_tokens + completion.usage.output_tokens
            except Exception as e:
                logging.error(f"{e}\n{traceback.format_exc()}")
                raise ApiRequestException

        try:
            tokens_count = 0
            text = ""
            with self.client.messages.stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
            ) as stream:
                empty_stream = True
                error = False
                for event in stream:
                    empty_stream = False
                    name = event.__class__.__name__
                    if name == "MessageStartEvent":
                        if event.message.usage:
                            tokens_count += event.message.usage.input_tokens
                        else:
                            error = True
                    elif name == "ContentBlockDeltaEvent":
                        text += event.delta.text
                    elif name == "MessageDeltaEvent":
                        tokens_count += event.usage.output_tokens
                    elif name == "Error":
                        logging.error(event.error.message)
                        raise ApiRequestException
                if empty_stream:
                    raise ApiRequestException("Empty stream object, please check your proxy connection!")
                if error:
                    raise ApiRequestException(text)
                if not text:
                    raise ApiRequestException("Empty text result, please check your prefill!")
            return text, tokens_count
        except Exception as e:
            logging.error(f"{e}\n{traceback.format_exc()}")
            raise ApiRequestException

    def get_answer(self, message, reply_msg, photo_base64):
        chat_name = utils.username_parser(message) if message.chat.title is None else message.chat.title
        if reply_msg:
            if self.dialog_history[-1]['content'] == reply_msg['content']:
                reply_msg = None

        msg_txt = message.text or message.caption
        if msg_txt is None:
            msg_txt = "I sent a photo"

        prompt = ""
        if any([random.randint(1, 30) == 1,  # Time is reminded of Humanotronic with a probability of 1/30
                int(time.time()) - self.last_time >= 3600,
                "врем" in msg_txt.lower(),
                "час" in msg_txt.lower()
                ]):
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
            answer, total_tokens = self.send_api_request(self.config.model,
                                                         dialog_buffer,
                                                         self.config.tokens_per_answer,
                                                         self.system,
                                                         self.config.temperature,
                                                         self.config.stream_mode)
        except ApiRequestException:
            return random.choice(self.config.prompts.errors)
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
        self.last_time = int(time.time())
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
            answer, total_tokens = self.send_api_request(self.config.model,
                                                         compressed_dialogue,
                                                         self.config.tokens_per_answer, None,
                                                         self.config.temperature,
                                                         self.config.stream_mode)
        except ApiRequestException:
            logging.error(f"Summarizing failed for chat {chat_name}!")
            self.dialogue_locker = False
            return

        logging.info(f"Summarizing completed for chat {chat_name}, "
                     f"{total_tokens} tokens were used")
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
