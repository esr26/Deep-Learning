import os
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import wavio
import joblib
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model

# Load Pre-trained Model
MODEL_PATH = "lstm_emotion_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Load the trained LSTM model
model = load_model(MODEL_PATH)

# Load the label encoder
encoder = joblib.load(LABEL_ENCODER_PATH)

# Extract Features from Audio
def extract_features(file_path, max_pad_length=130):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)


    if mfcc.shape[1] < max_pad_length:
        pad_width = max_pad_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_length]

    return mfcc.T  # Transpose to match LSTM input shape

# Streamlit App
st.title("🎙 Real-Time Speech Emotion Recognition with LSTM")

# Record Audio
def record_audio(duration=3, fs=44100, filename="recorded.wav"):
    st.write("🎙 Recording... Please speak!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    
    # Save as WAV file
    wavio.write(filename, recording, fs, sampwidth=2)
    st.write("✅ Recorded Successfully!")
    return filename

# Predict Emotion
def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features[:130, :], axis=0) 

    prediction = model.predict(features)
    emotion = encoder.inverse_transform([np.argmax(prediction)])[0]
    return emotion

# Streamlit UI Button
if st.button("🎙 Record and Predict Emotion"):
    recorded_file = record_audio()
    predicted_emotion = predict_emotion(recorded_file)
    st.subheader(f"🎭 Detected Emotion: **{predicted_emotion}**")
