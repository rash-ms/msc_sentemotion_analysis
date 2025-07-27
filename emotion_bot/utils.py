import joblib
import re
from keras.preprocessing.sequence import pad_sequences

# Load DL models and tokenizer
base_path = '../model_src_code/save_models/'

sentiment_model = joblib.load(f"{base_path}/sent_cnn_best_model.pkl")  # CNN expects padded sequences
emotion_model = joblib.load(f"{base_path}/emo_cnn_best_model.pkl")
tokenizer = joblib.load(f"{base_path}/dl_tokenizer.pkl")  # Used to convert text to sequence

# Make sure this maxlen matches what was used during training
MAXLEN = 100

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def predict_sentiment_emotion(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN)

    sentiment = sentiment_model.predict(padded)
    emotion = emotion_model.predict(padded)

    sentiment_label = sentiment.argmax(axis=1)[0]
    emotion_label = emotion.argmax(axis=1)[0]

    return sentiment_label, emotion_label
