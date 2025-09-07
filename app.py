# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load saved model
# model = joblib.load("best_model.pkl")

# st.title("🎵 Music Streams Prediction App")
# st.write("Predict the number of streams for a track based on its features.")

# # User inputs
# released_year = st.number_input("Released Year", min_value=1900, max_value=2100, value=2023)
# released_month = st.number_input("Released Month", min_value=1, max_value=12, value=6)
# released_day = st.number_input("Released Day", min_value=1, max_value=31, value=15)
# artist_count = st.number_input("Artist Count", min_value=1, value=1)
# bpm = st.number_input("Tempo (BPM)", min_value=40, max_value=250, value=120)

# music_tempo = st.selectbox("Music Tempo", ["Medium", "Fast", "Slow"])
# key = st.selectbox("Key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
# mode = st.selectbox("Mode", ["Major", "Minor"])

# danceability = st.slider("Danceability (%)", 0, 100, 50)
# valence = st.slider("Valence (%)", 0, 100, 50)
# energy = st.slider("Energy (%)", 0, 100, 50)
# acousticness = st.slider("Acousticness (%)", 0, 100, 50)
# instrumentalness = st.slider("Instrumentalness (%)", 0, 100, 0)
# liveness = st.slider("Liveness (%)", 0, 100, 20)
# speechiness = st.slider("Speechiness (%)", 0, 100, 5)

# # Collect features into DataFrame
# input_data = pd.DataFrame({
#     "released_year": [released_year],
#     "released_month": [released_month],
#     "released_day": [released_day],
#     "artist_count": [artist_count],
#     "bpm": [bpm],
#     "music_tempo": [music_tempo],
#     "key": [key],
#     "mode": [mode],
#     "danceability_%": [danceability],
#     "valence_%": [valence],
#     "energy_%": [energy],
#     "acousticness_%": [acousticness],
#     "instrumentalness_%": [instrumentalness],
#     "liveness_%": [liveness],
#     "speechiness_%": [speechiness]
# })

# # Predict button
# if st.button("Predict Streams"):
#     prediction = model.predict(input_data)[0]
#     st.success(f"🎶 Predicted Streams: {int(prediction):,}")


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model (Pipeline)
model = joblib.load("best_model.pkl")

st.title("🎵 Music Streams Prediction App")
st.write("Predict the number of streams for a track based on its features.")

# Get expected features from the trained model
try:
    expected_features = model.feature_names_in_
except AttributeError:
    st.error("❌ Model does not have 'feature_names_in_'. Check your training pipeline.")
    st.stop()

# Dictionary for user inputs
user_inputs = {}

# Collect inputs dynamically based on features
for feature in expected_features:
    if "year" in feature.lower():
        user_inputs[feature] = st.number_input("Released Year", min_value=1900, max_value=2100, value=2023)
    elif "month" in feature.lower():
        user_inputs[feature] = st.number_input("Released Month", min_value=1, max_value=12, value=6)
    elif "day" in feature.lower():
        user_inputs[feature] = st.number_input("Released Day", min_value=1, max_value=31, value=15)
    elif "artist_count" in feature.lower():
        user_inputs[feature] = st.number_input("Artist Count", min_value=1, value=1)
    elif "bpm" in feature.lower():
        user_inputs[feature] = st.number_input("Tempo (BPM)", min_value=40, max_value=250, value=120)
    elif "music_tempo" in feature.lower():
        user_inputs[feature] = st.selectbox("Music Tempo", ["Medium", "Fast", "Slow"])
    elif feature == "key":
        user_inputs[feature] = st.selectbox("Key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    elif feature == "mode":
        user_inputs[feature] = st.selectbox("Mode", ["Major", "Minor"])
    elif "%" in feature.lower():
        user_inputs[feature] = st.slider(feature, 0, 100, 50)
    else:
        user_inputs[feature] = st.text_input(feature, "")

# Convert to DataFrame with correct column order
input_data = pd.DataFrame([[user_inputs[f] for f in expected_features]], columns=expected_features)

# Predict button
if st.button("Predict Streams"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🎶 Predicted Streams: {int(prediction):,}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
