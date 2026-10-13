import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50

st.set_page_config(page_title="TripAdvisor Review Rating Predictor", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("sentiment_lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

st.title("🏨 Hotel Review Sentiment & Rating Predictor")
st.write("Enter a hotel review below to predict the customer rating (1 to 5 stars).")

user_input = st.text_area("Review Text:", placeholder="The room was clean, staff was courteous...")

if st.button("Predict Rating"):
    if not user_input.strip():
        st.warning("Please enter a valid review.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)
        prediction = model.predict(padded)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        # Adjust based on whether your training targets were 0-indexed or 1-indexed
        predicted_rating = predicted_class + 1 

        st.success(f"**Predicted Rating:** {predicted_rating} / 5 Stars ⭐")
        st.bar_chart(prediction[0])
