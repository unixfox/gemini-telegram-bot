from __future__ import annotations
import datetime
import logging
import os
import base64

import openai

import json
import httpx
import io
from PIL import Image

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from utils import is_direct_result, encode_image, decode_image
from plugin_manager import PluginManager

# Models can be found here: https://platform.openai.com/docs/models/overview
# Models gpt-3.5-turbo-0613 and  gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
GPT_3_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613")
GPT_3_16K_MODELS = ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125")
GPT_4_MODELS = ("gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-turbo-preview")
GPT_4_32K_MODELS = ("gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613")
GPT_4_VISION_MODELS = ("gpt-4o",)
GPT_4_128K_MODELS = ("gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-turbo-2024-04-09")
GPT_4O_MODELS = ("gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest")
O_MODELS = ("o1", "o1-mini", "o1-preview")
GPT_ALL_MODELS = GPT_3_MODELS + GPT_3_16K_MODELS + GPT_4_MODELS + GPT_4_32K_MODELS + GPT_4_VISION_MODELS + GPT_4_128K_MODELS + GPT_4O_MODELS + O_MODELS

def default_max_tokens(model: str) -> int:
    """
    Gets the default number of max tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    return None


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    if model in ("gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-32k-0314", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"):
        return False
    if model in O_MODELS:
        return False
    return True


# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        # Fallback to English if the translation is not available
        if key in translations['en']:
            return translations['en'][key]
        else:
            logging.warning(f"No english definition found for key '{key}' in translations.json")
            # return key as text
            return key


class OpenAIHelper:
    """
    ChatGPT helper class.
    """

    def __init__(self, config: dict, plugin_manager: PluginManager):
        """
        Initializes the OpenAI helper class with the given configuration.
        :param config: A dictionary containing the GPT configuration
        :param plugin_manager: The plugin manager
        """
        http_client = httpx.AsyncClient(proxy=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)
        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int: list] = {}  # {chat_id: history}
        self.conversations_vision: dict[int: bool] = {}  # {chat_id: is_vision}
        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), 0

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, int]:
        """
        Gets a full response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query)
        if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(chat_id, response)
            if is_direct_result(response):
                return response, 0

        answer = ''

        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = choice.message.content.strip()
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            answer = response.choices[0].message.content.strip()
            self.__add_to_history(chat_id, role="assistant", content=answer)

        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, 0

    async def get_chat_response_stream(self, chat_id: int, query: str):
        """
        Stream response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used, or 'not_finished'
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query, stream=True)
        if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
            if is_direct_result(response):
                yield response, 0
                return

        answer = ''
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)

        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 Usage statistics not available for this model"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, 0

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            self.__add_to_history(chat_id, role="user", content=query)

            # Summarize the chat history if it's too long
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.__add_to_history(chat_id, role="user", content=query)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            common_args = {
                'model': self.config['model'] if not self.conversations_vision[chat_id] else self.config['vision_model'],
                'messages': self.conversations[chat_id],
                'temperature': self.config['temperature'],
                'n': self.config['n_choices'],
                'stream': stream
            }

            if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
                functions = self.plugin_manager.get_functions_specs()
                if len(functions) > 0:
                    common_args['functions'] = self.plugin_manager.get_functions_specs()
                    common_args['function_call'] = 'auto'

            return await self.client.chat.completions.create(**common_args)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def __handle_function_call(self, chat_id, response, stream=False, times=0, plugins_used=()):
        function_name = ''
        arguments = ''
        if stream:
            async for item in response:
                if len(item.choices) > 0:
                    first_choice = item.choices[0]
                    if first_choice.delta and first_choice.delta.function_call:
                        if first_choice.delta.function_call.name:
                            function_name += first_choice.delta.function_call.name
                        if first_choice.delta.function_call.arguments:
                            arguments += first_choice.delta.function_call.arguments
                    elif first_choice.finish_reason and first_choice.finish_reason == 'function_call':
                        break
                    else:
                        return response, plugins_used
                else:
                    return response, plugins_used
        else:
            if len(response.choices) > 0:
                first_choice = response.choices[0]
                if first_choice.message.function_call:
                    if first_choice.message.function_call.name:
                        function_name += first_choice.message.function_call.name
                    if first_choice.message.function_call.arguments:
                        arguments += first_choice.message.function_call.arguments
                else:
                    return response, plugins_used
            else:
                return response, plugins_used

        logging.info(f'Calling function {function_name} with arguments {arguments}')
        function_response = await self.plugin_manager.call_function(function_name, self, arguments)

        if function_name not in plugins_used:
            plugins_used += (function_name,)

        if is_direct_result(function_response):
            self.__add_function_call_to_history(chat_id=chat_id, function_name=function_name,
                                                content=json.dumps({'result': 'Done, the content has been sent'
                                                                              'to the user.'}))
            return function_response, plugins_used

        self.__add_function_call_to_history(chat_id=chat_id, function_name=function_name, content=function_response)
        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=self.conversations[chat_id],
            functions=self.plugin_manager.get_functions_specs(),
            function_call='auto' if times < self.config['functions_max_consecutive_calls'] else 'none',
            stream=stream
        )
        return await self.__handle_function_call(chat_id, response, stream, times + 1, plugins_used)

    async def generate_image(self, prompt: str) -> tuple[str, str]:
        """
        Generates an image from the given prompt using Gemini model.
        :param prompt: The prompt to send to the model
        :return: The image URL and the image size
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.images.generate(
                prompt=prompt,
                n=1,
                model=self.config['image_model'],
                quality=self.config['image_quality'],
                style=self.config['image_style'],
                size=self.config['image_size'],
                response_format='b64_json'
            )

            if not response or not response.data or len(response.data) == 0:
                logging.error(f'No response from Gemini: {str(response)}')
                raise Exception(
                    f"⚠️ {localized_text('error', bot_language)}. "
                    f"⚠️\n{localized_text('try_again', bot_language)}."
                )

            # For Gemini, the response might be in a different format
            # Let's try to handle both OpenAI and Gemini response formats
            image_data = None
            if hasattr(response.data[0], 'b64_json'):
                image_data = response.data[0].b64_json
            elif hasattr(response.data[0], 'data'):
                image_data = response.data[0].data
            elif isinstance(response.data[0], dict):
                image_data = response.data[0].get('b64_json') or response.data[0].get('data')
            
            if not image_data:
                raise Exception("No image data found in response")

            # Convert base64 to data URL for compatibility
            image_url = f"data:image/jpeg;base64,{image_data}"
            return image_url, self.config['image_size']
        except Exception as e:
            raise Exception(f"⚠️ {localized_text('error', bot_language)}. ⚠️\n{str(e)}") from e

    async def generate_speech(self, text: str) -> tuple[any, int]:
        """
        Generates an audio from the given text using TTS model.
        :param prompt: The text to send to the model
        :return: The audio in bytes
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.audio.speech.create(
                model=self.config['tts_model'],
                voice=self.config['tts_voice'],
                input=text,
                response_format='opus'
            )

            temp_file = io.BytesIO()
            temp_file.write(response.read())
            temp_file.seek(0)
            return temp_file, len(text)
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def transcribe(self, filename):
        """
        Transcribes the audio file using the Gemini model.
        """
        try:
            with open(filename, "rb") as audio:
                audio_bytes = audio.read()
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this audio"
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": filename.split('.')[-1].lower()
                            }
                        }
                    ]
                }
            ]

            response = await self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages
            )

            return response.choices[0].message.content
        except Exception as e:
            logging.exception(e)
            raise Exception(f"⚠️ _{localized_text('error', self.config['bot_language'])}._ ⚠️\n{str(e)}") from e

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response_vision(self, chat_id: int, content: list, stream=False):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            if self.config['enable_vision_follow_up_questions']:
                self.conversations_vision[chat_id] = True
                self.__add_to_history(chat_id, role="user", content=content)
            else:
                for message in content:
                    if message['type'] == 'text':
                        query = message['text']
                        break
                self.__add_to_history(chat_id, role="user", content=query)

            # Summarize the chat history if it's too long
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    last = self.conversations[chat_id][-1]
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.conversations[chat_id] += [last]
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            message = {'role':'user', 'content':content}

            common_args = {
                'model': self.config['vision_model'],
                'messages': self.conversations[chat_id][:-1] + [message],
                'temperature': self.config['temperature'],
                'n': 1,
                'stream': stream
            }

            return await self.client.chat.completions.create(**common_args)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e


    async def interpret_image(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type':'text', 'text':prompt}, {'type':'image_url', \
                    'image_url': {'url':image, 'detail':self.config['vision_detail'] } }]

        response = await self.__common_get_chat_response_vision(chat_id, content)

        answer = ''

        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = choice.message.content.strip()
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            answer = response.choices[0].message.content.strip()
            self.__add_to_history(chat_id, role="assistant", content=answer)

        return answer, 0

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type':'text', 'text':prompt}, {'type':'image_url', \
                    'image_url': {'url':image, 'detail':self.config['vision_detail'] } }]

        response = await self.__common_get_chat_response_vision(chat_id, content, stream=True)

        answer = ''
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)

        yield answer, 0

    def reset_chat_history(self, chat_id, content=''):
        """
        Resets the conversation history.
        """
        if content == '':
            content = self.config['assistant_prompt']
        self.conversations[chat_id] = [{"role": "assistant" if self.config['model'] in O_MODELS else "system", "content": content}]
        self.conversations_vision[chat_id] = False

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def __add_function_call_to_history(self, chat_id, function_name, content):
        """
        Adds a function call to the conversation history
        """
        self.conversations[chat_id].append({"role": "function", "name": function_name, "content": content})

    def __add_to_history(self, chat_id, role, content):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        """
        self.conversations[chat_id].append({"role": role, "content": content})

    async def __summarise(self, conversation) -> str:
        """
        Summarises the conversation history.
        :param conversation: The conversation history
        :return: The summary
        """
        messages = [
            {"role": "assistant", "content": "Summarize this conversation in 700 characters or less"},
            {"role": "user", "content": str(conversation)}
        ]
        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            temperature=1 if self.config['model'] in O_MODELS else 0.4
        )
        return response.choices[0].message.content

    def __max_model_tokens(self):
        base = 4096
        if self.config['model'] in GPT_3_MODELS:
            return base
        if self.config['model'] in GPT_3_16K_MODELS:
            return base * 4
        if self.config['model'] in GPT_4_MODELS:
            return base * 2
        if self.config['model'] in GPT_4_32K_MODELS:
            return base * 8
        if self.config['model'] in GPT_4_VISION_MODELS:
            return base * 31
        if self.config['model'] in GPT_4_128K_MODELS:
            return base * 31
        if self.config['model'] in GPT_4O_MODELS:
            return base * 31
        elif self.config['model'] in O_MODELS:
            # https://platform.openai.com/docs/models#o1
            if self.config['model'] == "o1":
                return 100_000
            elif self.config['model'] == "o1-preview":
                return 32_768
            else:
                return 65_536
        raise NotImplementedError(
            f"Max tokens for model {self.config['model']} is not implemented yet."
        )
