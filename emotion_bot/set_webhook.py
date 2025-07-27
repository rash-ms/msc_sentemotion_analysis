import os
import requests

BOT_TOKEN = os.getenv("BOT_TOKEN")
RENDER_URL = os.getenv("RENDER_URL")  # e.g., "https://your-app.onrender.com"

if not BOT_TOKEN or not RENDER_URL:
    raise ValueError("BOT_TOKEN and RENDER_URL must be set as environment variables.")

webhook_url = f"{RENDER_URL}/{BOT_TOKEN}"
set_webhook_url = f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook"

response = requests.post(set_webhook_url, params={"url": webhook_url})

if response.ok:
    print("Webhook set successfully:", response.json())
else:
    print("Failed to set webhook:", response.status_code, response.text)
