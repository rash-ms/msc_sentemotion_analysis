import joblib
import re

# Define base path
base_path = './save_models'

# Load trained models
sentiment_model = joblib.load(f"{base_path}/sent_1_naive_bayes_model.pkl")
emotion_model = joblib.load(f"{base_path}/emo_4_naive_bayes_model.pkl")

# Load vectorizers
sentiment_vectorizer = joblib.load(f"{base_path}/ml_sent_vectorizer.pkl")
emotion_vectorizer = joblib.load(f"{base_path}/ml_emo_vectorizer.pkl")

# Load saved label lists
sentiment_labels = joblib.load(f"{base_path}/ml_sent_vectorizer_labels.pkl")
emotion_labels = joblib.load(f"{base_path}/ml_emo_vectorizer_labels.pkl")

def clean_text(text):
    """Cleans input text by removing punctuation and lowering case."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def predict_sentiment_emotion(text, mode="both"):
    """Predicts sentiment and/or emotion from text and returns emoji-enhanced output."""
    text = clean_text(text)

    sentiment_emojis = {
        "positive": "😊",
        "negative": "😞",
        "neutral": "😐"
    }

    emotion_emojis = {
        "joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨",
        "surprise": "😲", "disgust": "🤢", "love": "❤️", "boredom": "😴", "neutral": "😐"
    }

    result_parts = []

    if mode in ["sentiment", "both"]:
        X_sent = sentiment_vectorizer.transform([text])
        sentiment_idx = sentiment_model.predict(X_sent)[0]
        sentiment = sentiment_labels[sentiment_idx]
        sent_icon = sentiment_emojis.get(sentiment.lower(), "🧠")
        result_parts.append(f"{sent_icon} Sentiment: {sentiment}")

    if mode in ["emotion", "both"]:
        X_emo = emotion_vectorizer.transform([text])
        emotion_idx = emotion_model.predict(X_emo)[0]
        emotion = emotion_labels[emotion_idx]
        emo_icon = emotion_emojis.get(emotion.lower(), "🎭")
        result_parts.append(f"{emo_icon} Emotion: {emotion}")

    return "\n".join(result_parts)
