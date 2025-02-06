# Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Mapping of the word index to words
word_index = imdb.get_word_index()
reverse_index = {val:key for key, val in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('rnn_imdb.h5')

# decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_index.get(i-3, '?') for i in encoded_review])

# preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Streamlit App
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a moview review to classify it as Positive or Negative.")

user_input = st.text_area('Moview Review')

if st.button('Predict'):
    preprocess_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:
    st.write("Please enter a movie review")

