import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies_df = pd.read_csv("movies.csv")

# Preprocess data (e.g., handle missing values, normalize ratings)

# Create user-item matrix
user_item_matrix = movies_df.pivot_table(values='release_year', index='show_id', columns='title')

# Calculate item similarities
item_similarities = cosine_similarity(user_item_matrix.fillna(0))

# Define recommendation function
def recommend_movies(movie_name, user_id, num_recommendations=10):
    movie_index = movies_df[movies_df['title'] == movie_name].index[0]
    similar_movies = item_similarities[movie_index]
    sorted_indexes = similar_movies.argsort()[::-1]
    recommendations = []
    for i in sorted_indexes:
        if user_item_matrix.iloc[user_id, i] == 0:  # Check if not already rated
            recommendations.append(movies_df.iloc[i]['title'])
            if len(recommendations) == num_recommendations:
                break
    return recommendations

# Get recommendations for a user
user_id = 1  # Replace with actual user ID
movie_name = "The Matrix"
recommendations = recommend_movies(movie_name, user_id)
print(recommendations)
