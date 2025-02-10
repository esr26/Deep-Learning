import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the LSTM model
model = load_model('next_word_lstm.h5')

# load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer= pickle.load(handle)


def predict_next_word(model, tokenizer, seed_text, max_sequence_len=6):
    """
    Predicts the next word given a seed text.

    Args:
        model (keras.Model): The trained LSTM model.
        tokenizer (keras.preprocessing.text.Tokenizer): The tokenizer used for text preprocessing.
        seed_text (str): The input text to predict the next word for.
        max_sequence_len (int): The maximum length of input sequences.

    Returns:
        str: The seed text with the predicted next word appended.
    """
    # Tokenize the seed text
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # Predict the next word
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]  # Get the index of the most likely word

    # Convert the index back to a word
    predicted_word = tokenizer.index_word.get(predicted_index, "<UNK>")  # Handle unknown words

    # Append the predicted word to the seed text
    return seed_text + " " + predicted_word

# Streamlit app
st.title("Next Word Prediction using LSTM")
input_text = st.text_input("Enter the sequence of Words")
if st.button("Predict Next Word"):
    predicted_text = predict_next_word(model, tokenizer, input_text)
    st.write(predicted_text)


