import joblib
import re
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# Save Path
base_path = './save_models'
# Load ML models
sentiment_model = joblib.load(f"{base_path}/sent_1_naive_bayes_model.pkl")
emotion_model = joblib.load(f"{base_path}/sent_4_naive_bayes_model.pkl")

# Load vectorizers
sentiment_vectorizer = joblib.load(f"{base_path}/ml_sent_vectorizer.pkl")
emotion_vectorizer = joblib.load(f"{base_path}/ml_emo_vectorizer.pkl")

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def predict_sentiment_emotion(text, mode="both"):
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

    if mode == "sentiment":
        X = sentiment_vectorizer.transform([text])
        sentiment_idx = sentiment_model.predict(X)[0]
        sentiment = sentiment_model.classes_[sentiment_idx]  # Convert numeric to label
        sent_icon = sentiment_emojis.get(sentiment.lower(), "🧠")
        return f"{sent_icon} Sentiment: {sentiment}"

    elif mode == "emotion":
        X = emotion_vectorizer.transform([text])
        emotion = emotion_model.predict(X)[0]
        emo_icon = emotion_emojis.get(str(emotion).lower(), "🎭")  # safer with str()
        return f"{emo_icon} Emotion: {emotion}"

    else:
        X_sent = sentiment_vectorizer.transform([text])
        sentiment_idx = sentiment_model.predict(X_sent)[0]
        sentiment = sentiment_model.classes_[sentiment_idx]
        sent_icon = sentiment_emojis.get(sentiment.lower(), "🧠")

        X_emo = emotion_vectorizer.transform([text])
        emotion = emotion_model.predict(X_emo)[0]
        emo_icon = emotion_emojis.get(str(emotion).lower(), "🎭")

        return f"{sent_icon} Sentiment: {sentiment}\n{emo_icon} Emotion: {emotion}"



# def predict_sentiment_emotion(text, mode="both"):
#     text = clean_text(text)
#
#     sentiment_emojis = {
#         "positive": "😊",
#         "negative": "😞",
#         "neutral": "😐"
#     }
#
#     emotion_emojis = {
#         "joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨",
#         "surprise": "😲", "disgust": "🤢", "love": "❤️", "boredom": "😴", "neutral": "😐"
#     }
#
#     if mode == "sentiment":
#         X = sentiment_vectorizer.transform([text])
#         sentiment = sentiment_model.predict(X)[0]
#         sent_icon = sentiment_emojis.get(sentiment.lower(), "🧠")
#         return f"{sent_icon} Sentiment: {sentiment}"
#
#     elif mode == "emotion":
#         X = emotion_vectorizer.transform([text])
#         emotion = emotion_model.predict(X)[0]
#         emo_icon = emotion_emojis.get(emotion.lower(), "🎭")
#         return f"{emo_icon} Emotion: {emotion}"
#
#     else:
#         X_sent = sentiment_vectorizer.transform([text])
#         sentiment = sentiment_model.predict(X_sent)[0]
#         sent_icon = sentiment_emojis.get(sentiment.lower(), "🧠")
#
#         X_emo = emotion_vectorizer.transform([text])
#         emotion = emotion_model.predict(X_emo)[0]
#         emo_icon = emotion_emojis.get(emotion.lower(), "🎭")
#
#         return f"{sent_icon} Sentiment: {sentiment}\n{emo_icon} Emotion: {emotion}"
#
