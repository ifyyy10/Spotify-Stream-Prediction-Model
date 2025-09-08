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


# import streamlit as st
# import pandas as pd
# import numpy as np
# import skops.io as sio

# # Load the trained model (skops version)
# @st.cache_resource
# def load_model():
#     # Step 1: Inspect untrusted types in the file
#     untrusted_types = sio.get_untrusted_types(file="best_model.skops")

#     # Step 2: Load the model, explicitly trusting these types
#     model = sio.load("best_model.skops", trusted=untrusted_types)
#     return model

# model = load_model()

# # 🎵 Title
# st.title("🎶 Spotify Streams Prediction App")

# st.markdown(
#     """
#     Upload or enter song features to predict the number of streams.
#     """
# )

# # === Sidebar for Mode Selection ===
# mode = st.sidebar.radio("Choose Prediction Mode:", ["Single Prediction", "Batch Prediction (CSV)"])

# # === Single Prediction Mode ===
# if mode == "Single Prediction":
#     st.header("Enter Song Features")

#     # Numeric features
#     numeric_features = {
#         "acousticness_%": st.number_input("Acousticness %", 0.0, 100.0, 50.0),
#         "artist_count": st.number_input("Artist Count", 1, 10, 1),
#         "bpm": st.number_input("Beats Per Minute (BPM)", 40, 250, 120),
#         "danceability_%": st.number_input("Danceability %", 0.0, 100.0, 50.0),
#         "energy_%": st.number_input("Energy %", 0.0, 100.0, 50.0),
#         "in_apple_charts": st.number_input("In Apple Charts", 0, 2000, 0),
#         "in_apple_playlists": st.number_input("In Apple Playlists", 0, 2000, 0),
#         "in_deezer_charts": st.number_input("In Deezer Charts", 0, 2000, 0),
#         "in_deezer_playlists": st.number_input("In Deezer Playlists", 0, 2000, 0),
#         "in_shazam_charts": st.number_input("In Shazam Charts", 0, 2000, 0),
#         "in_spotify_charts": st.number_input("In Spotify Charts", 0, 2000, 0),
#         "in_spotify_playlists": st.number_input("In Spotify Playlists", 0, 2000, 0),
#         "instrumentalness_%": st.number_input("Instrumentalness %", 0.0, 100.0, 0.0),
#         "liveness_%": st.number_input("Liveness %", 0.0, 100.0, 50.0),
#         "released_day": st.number_input("Release Day", 1, 31, 1),
#         "released_month": st.number_input("Release Month", 1, 12, 1),
#         "released_year": st.number_input("Release Year", 1950, 2030, 2023),
#         "speechiness_%": st.number_input("Speechiness %", 0.0, 100.0, 50.0),
#         "valence_%": st.number_input("Valence %", 0.0, 100.0, 50.0),
#     }

#     # Categorical features
#     st.subheader("Categorical Features")
#     music_tempo = st.selectbox("Music Tempo", ["Slow", "Medium", "Fast"])
#     key = st.selectbox("Key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
#     mode_input = st.selectbox("Mode", ["Major", "Minor"])

#     # Prepare input
#     input_data = pd.DataFrame([{
#         **numeric_features,
#         "music_tempo": music_tempo,
#         "key": key,
#         "mode": mode_input
#     }])

#     # Prediction button
#     if st.button("Predict Streams"):
#         prediction = model.predict(input_data)[0]
#         st.success(f"🎧 Predicted Streams: {int(prediction):,}")

# # === Batch Prediction Mode ===
# else:
#     st.header("Batch Prediction via CSV Upload")
#     st.markdown("Upload a CSV file with the same feature columns used in training.")

#     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)

#         st.write("📊 Uploaded Data Preview:", df.head())

#         try:
#             predictions = model.predict(df)
#             df["Predicted_Streams"] = predictions.astype(int)

#             st.success("✅ Predictions generated successfully!")
#             st.write(df.head())

#             # Download option
#             csv_out = df.to_csv(index=False).encode("utf-8")
#             st.download_button(
#                 label="📥 Download Predictions",
#                 data=csv_out,
#                 file_name="predicted_streams.csv",
#                 mime="text/csv",
#             )
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")


import streamlit as st
import pandas as pd
import numpy as np
import skops.io as sio

# Load the trained model (skops version)
@st.cache_resource
def load_model():
    # Step 1: Inspect untrusted types in the file
    untrusted_types = sio.get_untrusted_types(file="best_model.skops")

    # Step 2: Load the model, explicitly trusting these types
    model = sio.load("best_model.skops", trusted=untrusted_types)

    # 🚑 Patch: Remove GPU-related parameters if they exist
    try:
        params = model.get_params()
        if "gpu_id" in params:
            params.pop("gpu_id", None)
            model.set_params(**params)
        if "tree_method" in params and params["tree_method"] == "gpu_hist":
            model.set_params(tree_method="hist", predictor="cpu_predictor")
    except Exception as e:
        st.warning(f"⚠️ Model patching skipped: {e}")

    return model

model = load_model()

# 🎵 Title
st.title("🎶 Spotify Streams Prediction App")

st.markdown(
    """
    Upload or enter song features to predict the number of streams.
    """
)

# === Sidebar for Mode Selection ===
mode = st.sidebar.radio("Choose Prediction Mode:", ["Single Prediction", "Batch Prediction (CSV)"])

# === Single Prediction Mode ===
if mode == "Single Prediction":
    st.header("Enter Song Features")

    # Numeric features
    numeric_features = {
        "acousticness_%": st.number_input("Acousticness %", 0.0, 100.0, 50.0),
        "artist_count": st.number_input("Artist Count", 1, 10, 1),
        "bpm": st.number_input("Beats Per Minute (BPM)", 40, 250, 120),
        "danceability_%": st.number_input("Danceability %", 0.0, 100.0, 50.0),
        "energy_%": st.number_input("Energy %", 0.0, 100.0, 50.0),
        "in_apple_charts": st.number_input("In Apple Charts", 0, 2000, 0),
        "in_apple_playlists": st.number_input("In Apple Playlists", 0, 2000, 0),
        "in_deezer_charts": st.number_input("In Deezer Charts", 0, 2000, 0),
        "in_deezer_playlists": st.number_input("In Deezer Playlists", 0, 2000, 0),
        "in_shazam_charts": st.number_input("In Shazam Charts", 0, 2000, 0),
        "in_spotify_charts": st.number_input("In Spotify Charts", 0, 2000, 0),
        "in_spotify_playlists": st.number_input("In Spotify Playlists", 0, 2000, 0),
        "instrumentalness_%": st.number_input("Instrumentalness %", 0.0, 100.0, 0.0),
        "liveness_%": st.number_input("Liveness %", 0.0, 100.0, 50.0),
        "released_day": st.number_input("Release Day", 1, 31, 1),
        "released_month": st.number_input("Release Month", 1, 12, 1),
        "released_year": st.number_input("Release Year", 1950, 2030, 2023),
        "speechiness_%": st.number_input("Speechiness %", 0.0, 100.0, 50.0),
        "valence_%": st.number_input("Valence %", 0.0, 100.0, 50.0),
    }

    # Categorical features
    st.subheader("Categorical Features")
    music_tempo = st.selectbox("Music Tempo", ["Slow", "Medium", "Fast"])
    key = st.selectbox("Key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    mode_input = st.selectbox("Mode", ["Major", "Minor"])

    # Prepare input
    input_data = pd.DataFrame([{
        **numeric_features,
        "music_tempo": music_tempo,
        "key": key,
        "mode": mode_input
    }])

    # Prediction button
    if st.button("Predict Streams"):
        prediction = model.predict(input_data)[0]
        st.success(f"🎧 Predicted Streams: {int(prediction):,}")

# === Batch Prediction Mode ===
else:
    st.header("Batch Prediction via CSV Upload")
    st.markdown("Upload a CSV file with the same feature columns used in training.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("📊 Uploaded Data Preview:", df.head())

        try:
            predictions = model.predict(df)
            df["Predicted_Streams"] = predictions.astype(int)

            st.success("✅ Predictions generated successfully!")
            st.write(df.head())

            # Download option
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions",
                data=csv_out,
                file_name="predicted_streams.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")

