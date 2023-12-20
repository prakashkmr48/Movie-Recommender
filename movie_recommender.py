import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (replace with your actual path)
movies_df = pd.read_csv("movies.csv")

# Preprocess data (e.g., handle missing values, normalize ratings)

# Create user-item matrix
user_item_matrix = movies_df.pivot_table(values='rating', index='userId', columns='movieId')

# Calculate item similarities
item_similarities = cosine_similarity(user_item_matrix.fillna(0))

# Define recommendation function
def recommend_movies(movie_name, user_id, num_recommendations=10):
    # ... (same as before)

# Create Streamlit app layout
st.title("Movie Recommender")

# Get user input
user_id = st.number_input("Enter your user ID", value=1)
movie_name = st.text_input("Enter a movie you like")

# Generate and display recommendations
if movie_name:
    recommendations = recommend_movies(movie_name, user_id)
    st.write("Recommended movies for you:")
    st.write(recommendations)
