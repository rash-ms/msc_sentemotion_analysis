from flask import Flask, request
import requests
from utils import predict_sentiment_emotion

app = Flask(__name__)
BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'
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

        sentiment, emotion = predict_sentiment_emotion(text)
        reply = f"🧠 *Sentiment*: {sentiment}\n🎭 *Emotion*: {emotion}"

        send_message(chat_id, reply)

    return {'ok': True}

def send_message(chat_id, text):
    requests.post(f"{TELEGRAM_URL}/sendMessage", json={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    })

if __name__ == '__main__':
    app.run(debug=True)
