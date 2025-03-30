import os.path
import pathlib
import json
from datetime import date


def year_month(date_str):
    # extract string of year-month from date, eg: '2023-03'
    return str(date_str)[:7]


class UsageTracker:
    """
    UsageTracker class
    Enables tracking of daily/monthly usage per user.
    User files are stored as JSON in /usage_logs directory.
    JSON example:
    {
        "user_name": "@user_name",
        "current_cost": {
            "day": 0.45,
            "month": 3.23,
            "all_time": 3.23,
            "last_update": "2023-03-14"},
        "usage_history": {
            "chat_tokens": {
                "2023-03-13": 520,
                "2023-03-14": 1532
            },
            "transcription_seconds": {
                "2023-03-13": 125,
                "2023-03-14": 64
            },
            "number_images": {
                "2023-03-12": [0, 2, 3],
                "2023-03-13": [1, 2, 3],
                "2023-03-14": [0, 1, 2]
            }
        }
    }
    """

    def __init__(self, user_id, user_name, logs_dir="usage_logs"):
        """
        Initializes UsageTracker for a user with current date.
        Loads usage data from usage log file.
        :param user_id: Telegram ID of the user
        :param user_name: Telegram user name
        :param logs_dir: path to directory of usage logs, defaults to "usage_logs"
        """
        self.user_id = user_id
        self.logs_dir = logs_dir
        # path to usage file of given user
        self.user_file = f"{logs_dir}/{user_id}.json"

        if os.path.isfile(self.user_file):
            with open(self.user_file, "r") as file:
                self.usage = json.load(file)
            if 'vision_tokens' not in self.usage['usage_history']:
                self.usage['usage_history']['vision_tokens'] = {}
            if 'tts_characters' not in self.usage['usage_history']:
                self.usage['usage_history']['tts_characters'] = {}
        else:
            # ensure directory exists
            pathlib.Path(logs_dir).mkdir(exist_ok=True)
            # create new dictionary for this user
            self.usage = {
                "user_name": user_name,
                "current_cost": {"day": 0.0, "month": 0.0, "all_time": 0.0, "last_update": str(date.today())},
                "usage_history": {"number_images": {}, "tts_characters": {}, "vision_tokens":{}}
            }

    def get_current_token_usage(self):
        """Get token amounts used for today and this month
        :return: total number of tokens used per day and per month
        """
        return 0, 0

    def add_chat_tokens(self, tokens, tokens_price=0.002):
        """Placeholder for compatibility"""
        pass

    def get_current_image_count(self):
        """Get number of images requested for today and this month.
        :return: total number of images requested per day and per month
        """
        today = date.today()
        if str(today) in self.usage["usage_history"]["number_images"]:
            usage_day = sum(self.usage["usage_history"]["number_images"][str(today)])
        else:
            usage_day = 0
        month = str(today)[:7]  # year-month as string
        usage_month = 0
        for today, images in self.usage["usage_history"]["number_images"].items():
            if today.startswith(month):
                usage_month += sum(images)
        return usage_day, usage_month

    def get_current_cost(self):
        """Get total USD amount of all requests of the current day and month
        :return: cost of current day and month
        """
        today = date.today()
        last_update = date.fromisoformat(self.usage["current_cost"]["last_update"])
        if today == last_update:
            cost_day = self.usage["current_cost"]["day"]
            cost_month = self.usage["current_cost"]["month"]
        else:
            cost_day = 0.0
            if today.month == last_update.month:
                cost_month = self.usage["current_cost"]["month"]
            else:
                cost_month = 0.0
        cost_all_time = self.usage["current_cost"].get("all_time", self.initialize_all_time_cost())
        return {"cost_today": cost_day, "cost_month": cost_month, "cost_all_time": cost_all_time}

    def initialize_all_time_cost(self, image_prices="0.016,0.018,0.02", minute_price=0.006, vision_token_price=0.01, tts_prices='0.015,0.030'):
        """Get total USD amount of all requests in history
        :return: total cost of all requests
        """
        total_images = [sum(values) for values in zip(*self.usage['usage_history']['number_images'].values())]
        image_prices_list = [float(x) for x in image_prices.split(',')]
        image_cost = sum([count * price for count, price in zip(total_images, image_prices_list)])

        total_vision_tokens = sum(self.usage['usage_history']['vision_tokens'].values())
        vision_cost = round(total_vision_tokens * vision_token_price / 1000, 2)

        total_characters = [sum(tts_model.values()) for tts_model in self.usage['usage_history']['tts_characters'].values()]
        tts_prices_list = [float(x) for x in tts_prices.split(',')]
        tts_cost = round(sum([count * price / 1000 for count, price in zip(total_characters, tts_prices_list)]), 2)

        all_time_cost = image_cost + vision_cost + tts_cost
        return all_time_cost
