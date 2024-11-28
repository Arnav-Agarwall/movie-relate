from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess the dataset
movies_data = pd.read_csv('tmdb_5000_movies.csv')

# Parse genres column
movies_data['parsed_genres'] = movies_data['genres'].apply(
    lambda x: [genre['name'] for genre in ast.literal_eval(x)] if pd.notnull(x) else []
)

# Combine genres into a single string for vectorization
movies_data['genre_string'] = movies_data['parsed_genres'].apply(lambda x: ' '.join(x))

# Create a genre-based vector representation using CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary=True)
genre_matrix = vectorizer.fit_transform(movies_data['genre_string'])

# Function to recommend top N movies based on cosine similarity
def recommend_by_movie_name(movie_name, top_n=10):
    # Check if the movie exists in the dataset
    if movie_name not in movies_data['title'].values:
        return None, f"Movie '{movie_name}' not found in the dataset."
    
    # Get the index of the input movie
    movie_index = movies_data[movies_data['title'] == movie_name].index[0]
    
    # Compute cosine similarity between the input movie and all other movies
    cosine_sim = cosine_similarity(genre_matrix[movie_index], genre_matrix).flatten()
    
    # Get similarity scores with their indices, excluding the input movie
    similarity_scores = list(enumerate(cosine_sim))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Select the top N movies, excluding the input movie itself
    top_similar_movies = [
        movies_data.iloc[i][['title', 'vote_average', 'popularity']].to_dict()
        for i, score in similarity_scores[1:top_n + 1]
    ]
    
    return top_similar_movies, None

# Define API route
@app.route('/recommend', methods=['GET'])
def recommend():
    # Get the movie name from query parameters
    movie_name = request.args.get('movie_name')
    if not movie_name:
        return jsonify({"error": "Please provide a 'movie_name' parameter."}), 400
    
    # Get recommendations
    recommendations, error = recommend_by_movie_name(movie_name)
    if error:
        return jsonify({"error": error}), 404
    
    # Return recommendations as JSON
    return jsonify({"movie_name": movie_name, "recommendations": recommendations})

# Run the Flask app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
