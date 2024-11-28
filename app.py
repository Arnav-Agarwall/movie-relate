import streamlit as st
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load and preprocess the dataset
@st.cache_data
def load_data():
    # Load the dataset
    movies_data = pd.read_csv('tmdb_5000_movies.csv')
    
    # Parse genres column
    movies_data['parsed_genres'] = movies_data['genres'].apply(
        lambda x: [genre['name'] for genre in ast.literal_eval(x)] if pd.notnull(x) else []
    )
    
    # Combine genres into a single string for vectorization
    movies_data['genre_string'] = movies_data['parsed_genres'].apply(lambda x: ' '.join(x))
    
    return movies_data

movies_data = load_data()

# Create a genre-based vector representation
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary=True, token_pattern=None)
genre_matrix = vectorizer.fit_transform(movies_data['genre_string'])

# Function to recommend movies
def recommend_by_movie_name(movie_name, top_n=10):
    if movie_name not in movies_data['title'].values:
        return None, f"Movie '{movie_name}' not found in the dataset."
    
    # Get the index of the input movie
    movie_index = movies_data[movies_data['title'] == movie_name].index[0]
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(genre_matrix[movie_index], genre_matrix).flatten()
    
    # Get similarity scores with their indices, excluding the input movie
    similarity_scores = list(enumerate(cosine_sim))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Select the top N movies, excluding the input movie itself
    recommendations = [
        movies_data.iloc[i][['title', 'vote_average', 'popularity']].to_dict()
        for i, score in similarity_scores[1:top_n + 1]
    ]
    
    return recommendations, None

# Streamlit UI
st.title("Movie Recommendation System 🎥")

# Input for movie name
movie_name = st.text_input("Enter a movie name", "")

if st.button("Get Recommendations"):
    if not movie_name.strip():
        st.error("Please enter a valid movie name!")
    else:
        recommendations, error = recommend_by_movie_name(movie_name)
        if error:
            st.error(error)
        else:
            st.success(f"Recommendations based on '{movie_name}':")
            for idx, movie in enumerate(recommendations, start=1):
                st.write(
                    f"{idx}. **{movie['title']}** - Rating: {movie['vote_average']}, Popularity: {movie['popularity']}"
                )
