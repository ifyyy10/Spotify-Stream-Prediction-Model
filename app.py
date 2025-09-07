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
import cloudpickle

# Load the saved model
with open("best_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

st.title("🎵 Spotify Stream Prediction App")

# Example UI
st.write("Fill in the track details below to predict streams:")

# Input widgets (example subset)
acousticness = st.slider("Acousticness (%)", 0.0, 100.0, 50.0)
danceability = st.slider("Danceability (%)", 0.0, 100.0, 50.0)
energy = st.slider("Energy (%)", 0.0, 100.0, 50.0)
key = st.selectbox("Key", options=list(range(0, 12)))
mode = st.selectbox("Mode", options=[0, 1])
tempo = st.selectbox("Tempo", options=["Fast", "Medium", "Slow"])  # depends on encoding

if st.button("Predict Streams"):
    # Convert inputs into dataframe (must match training columns!)
    input_df = pd.DataFrame([{
        "acousticness_%": acousticness,
        "danceability_%": danceability,
        "energy_%": energy,
        "key": key,
        "mode": mode,
        "music_tempo": tempo,
        # add other required features...
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"🎶 Predicted Streams: {int(prediction):,}")

