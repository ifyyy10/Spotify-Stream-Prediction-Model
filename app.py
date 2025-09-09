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

#     # 🚑 Patch: Remove GPU-related parameters if they exist
#     try:
#         params = model.get_params()
#         if "gpu_id" in params:
#             params.pop("gpu_id", None)
#             model.set_params(**params)
#         if "tree_method" in params and params["tree_method"] == "gpu_hist":
#             model.set_params(tree_method="hist", predictor="cpu_predictor")
#     except Exception as e:
#         st.warning(f"⚠️ Model patching skipped: {e}")

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

    # Define expected columns (same as training)
    expected_columns = [
        "acousticness_%", "artist_count", "bpm", "danceability_%", "energy_%",
        "in_apple_charts", "in_apple_playlists", "in_deezer_charts", "in_deezer_playlists",
        "in_shazam_charts", "in_spotify_charts", "in_spotify_playlists",
        "instrumentalness_%", "liveness_%", "released_day", "released_month",
        "released_year", "speechiness_%", "valence_%", "music_tempo", "key", "mode"
    ]

    # 📑 Sample row for template
    sample_data = {
        "acousticness_%": 45.0,
        "artist_count": 1,
        "bpm": 120,
        "danceability_%": 65.0,
        "energy_%": 70.0,
        "in_apple_charts": 10,
        "in_apple_playlists": 50,
        "in_deezer_charts": 5,
        "in_deezer_playlists": 30,
        "in_shazam_charts": 20,
        "in_spotify_charts": 15,
        "in_spotify_playlists": 200,
        "instrumentalness_%": 0.0,
        "liveness_%": 30.0,
        "released_day": 12,
        "released_month": 8,
        "released_year": 2022,
        "speechiness_%": 55.0,
        "valence_%": 60.0,
        "music_tempo": "Medium",
        "key": "C#",
        "mode": "Major"
    }

    # 📥 Downloadable template CSV (with sample row)
    template_df = pd.DataFrame([sample_data])
    st.download_button(
        label="📑 Download CSV Template (with sample row)",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="spotify_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:", df.head())

        # ✅ Check for missing and extra columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_columns]

        if missing_cols:
            st.error(f"❌ Wrong CSV format. Missing columns: {missing_cols}")
        else:
            if extra_cols:
                st.warning(f"⚠️ Extra columns found: {extra_cols}. They will be ignored.")
                df = df.drop(columns=extra_cols)

            # ✅ Reorder columns to match training order
            df = df[expected_columns]

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

