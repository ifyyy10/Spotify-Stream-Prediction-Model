Spotify Data Analysis Project

Overview
This project performs comprehensive analysis and predictive modeling on Spotify's top songs dataset for 2023. The work includes data cleaning, exploratory data analysis (EDA), feature engineering, building machine learning models to predict song popularity, and a Streamlit web application for users to interact with the model.

Project Structure
The analysis is organized in a Kaggle notebook (my-spotify-analyse.ipynb) with the following main sections:


1. Data Loading and Initial Exploration
Loads the Spotify 2023 dataset with proper encoding handling

Displays basic dataset information (shape, columns, data types)

Shows the first few rows of data for initial inspection


2. Data Cleaning and Preprocessing
Identifies and handles missing values in key columns

Converts appropriate columns to proper numeric data types

Creates a cleaned dataset copy for analysis

Checks for and removes duplicate entries

Identifies and handles potential outliers in various features


3. Feature Engineering
Creates new features including:

release_date: Combined date feature from year, month, day

days_since_release: Days since song release

streams_per_day: Daily streaming rate

music_tempo: Categorical tempo classification (slow, medium, upbeat, fast)


4. Exploratory Data Analysis
Analyzes top artists by total streams and most efficient artist per song streams

Examines relationships between musical features and popularity

Visualizes trends in the data


5. Predictive Modeling
Builds multiple regression models to predict song streams

Implements hyperparameter tuning with Optuna

Evaluates model performance using various metrics

Compares different algorithms and selects the best performer


6. Streamlit Web Application
Interactive Interface: User-friendly GUI for model testing

Input Controls: Sliders, dropdowns, and input fields for song features

Real-time Predictions: Instant streaming count predictions based on user inputs

Visual Feedback: Clear presentation of prediction results


Key Findings
The analysis reveals:

Top streaming artists on Spotify in 2023

Top streaming songs (most efficient artists) on Spotify in 2023

Trends in release patterns and their impact on popularity

Relationships between musical characteristics, artist colabo, release dates and streaming performance

The most effective machine learning model for predicting song popularity

Key factors that contribute to a song's success on Spotify


Technologies Used
Python 3.11.13

Core Libraries: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn, xgboost, lightgbm

Hyperparameter Tuning: optuna

Saved Model Format: joblib, skops

Web Framework: streamlit


Installation Requirements
bash
pip install -U scikit-learn>=1.4.0
pip install -U xgboost>=2.0.0
pip install -U lightgbm
pip install -U skops
pip install optuna


Installation & Setup
Clone the repository:
bash
git clone <your-repo-url>
cd Spotify-Stream-Prediction-Model

Create a virtual environment (recommended):
bash
python -m venv venv
source venv/bin/activate  (On Windows: venv\Scripts\activate)

Install dependencies:
bash
pip install -r requirements.txt


Dataset
The analysis uses the "Top Spotify Songs 2023" dataset from Kaggle, containing 953 songs with 24 features including:

Track metadata or Identification (name, artists, release date)

Release Information (released_year, released_month, released_day)

Platform performance metrics (in_spotify_playlists, in_spotify_charts, streams, in_apple_playlists, in_apple_charts, in_deezer_playlists, in_deezer_charts, in_shazam_charts)

Audio features (bpm, key, mode, danceability, valence, energy, acousticness, instrumentalness, liveness, speechiness)


Key Features
Comprehensive Data Cleaning: Handles missing values, data type conversions, and outlier detection

Feature Engineering: Creates meaningful new features from existing data

Visual Analysis: Explores relationships between musical characteristics and popularity

Multiple Modeling Approaches: Tests various regression algorithms

Hyperparameter Optimization: Uses Optuna for efficient parameter tuning

Model Evaluation: Compares models using MAE, RMSE, and R² metrics


Usage
To run this analysis:

Ensure you have Python 3.11+ installed

Install required packages (listed above)

Open the Kaggle notebook and run cells sequentially

The dataset should be placed in the /kaggle/input/top-spotify-songs-2023/ directory


Future Work
Potential extensions of this analysis could include:

Time series analysis of music trends

Genre-specific predictive modeling

Recommendation system development

Sentiment analysis of song lyrics

Comparison with previous years' data to identify trends


Author
[Ifeanyichukwu Okafor/ifyyy10]


License
This project is open source 
