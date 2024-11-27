from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import ast

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess the dataset
movies_data = pd.read_csv('tmdb_5000_movies.csv')

# Parse genres column
movies_data['parsed_genres'] = movies_data['genres'].apply(
    lambda x: [genre['name'] for genre in ast.literal_eval(x)] if pd.notnull(x) else []
)

# Function to recommend top 10 movies based on genres of an input movie
def recommend_by_movie_name(movie_name, top_n=10):
    # Check if the movie exists in the dataset
    if movie_name not in movies_data['title'].values:
        return None, f"Movie '{movie_name}' not found in the dataset."
    
    # Get genres of the input movie
    input_movie_genres = movies_data[movies_data['title'] == movie_name]['parsed_genres'].values[0]
    
    # Filter movies that share at least one genre with the input movie
    filtered_movies = movies_data[movies_data['parsed_genres'].apply(
        lambda genres: any(genre in input_movie_genres for genre in genres)
    )]
    
    # Sort movies by rating (vote_average), breaking ties with popularity
    sorted_movies = filtered_movies.sort_values(
        by=['vote_average', 'popularity'], ascending=[False, False]
    )
    
    # Exclude the input movie from the recommendations
    sorted_movies = sorted_movies[sorted_movies['title'] != movie_name]
    
    # Select the top N movies
    top_movies = sorted_movies[['title', 'vote_average', 'popularity']].head(top_n)
    return top_movies.to_dict(orient='records'), None

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