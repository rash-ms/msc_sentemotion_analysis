import os
from flask import Flask, request
import requests
from utils import predict_sentiment_emotion

app = Flask(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

@app.route('/', methods=['GET'])
def home():
    return "Bot running!"

@app.route(f"/{BOT_TOKEN}", methods=['POST'])
def telegram_webhook():
    data = request.get_json()

    if 'message' in data:
        chat_id = data['message']['chat']['id']
        text = data['message'].get('text', '')
        message_id = data['message']['message_id']

        # reply = predict_sentiment_emotion(text)
        reply = predict_sentiment_emotion(text, mode="both")
        send_message(chat_id, reply, reply_to=message_id)

    return {'ok': True}


def send_message(chat_id, text, reply_to=None):
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }

    if reply_to:
        payload["reply_to_message_id"] = reply_to  # Reply formatting

    requests.post(f"{TELEGRAM_URL}/sendMessage", json=payload)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
