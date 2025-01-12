from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import requests
import re

app = Flask(__name__)
CORS(app)

# Load pre-trained model components
with open('./users_latent_without_val_and_timestamp.pkl', 'rb') as file:
    users_latent = pickle.load(file)
with open('./movies_latent_without_val_and_timestamp.pkl', 'rb') as file:
    movies_latent = pickle.load(file)
with open('./users_biases_without_val_and_timestamp.pkl', 'rb') as file:
    users_biases = pickle.load(file)
with open('./movies_biases_without_val_and_timestamp.pkl', 'rb') as file:
    movies_biases = pickle.load(file)
with open('./idx_to_movie.pkl', 'rb') as file:
    idx_to_movie = pickle.load(file)
with open('./movie_to_idx.pkl', 'rb') as file:
    movie_to_idx = pickle.load(file)

# with open('./user_latent.pkl', 'rb') as file:
#     users_latent = pickle.load(file)
# with open('./item_latent.pkl', 'rb') as file:
#     movies_latent = pickle.load(file)
# with open('./user_bias.pkl', 'rb') as file:
#     users_biases = pickle.load(file)
# with open('./item_bias.pkl', 'rb') as file:
#     movies_biases = pickle.load(file)


lambd = 0.5
tau = 0.5
gamma = 0.1
factors = 32


movies = pd.read_csv("./movies.csv")
titles = movies['title'].values

# TMDb API Configuration
TMDB_API_KEY = '918cce94627c68fa3cb45b04c4dc0691'  # Replace with your TMDb API Key
TMDB_SEARCH_URL = 'https://api.themoviedb.org/3/search/movie'
TMDB_MOVIE_DETAILS_URL = 'https://api.themoviedb.org/3/movie/'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'


def calculate_dummy_user_bias(user_dummy, iterations, dummy_user_latent):
    bias_sum = 0
    item_counter = 0
    for movie_id, rating in user_dummy:
        movie_index = movie_to_idx[movie_id]
        if iterations == 0:
            bias_sum += lambd * (rating - movies_biases[movie_index])
        else:
            bias_sum += lambd * (rating - (
                np.dot(dummy_user_latent.T, movies_latent[:, movie_index]) + movies_biases[movie_index]))
        item_counter += 1
    return bias_sum / ((lambd * item_counter) + gamma) if item_counter > 0 else 0


def update_user_latent_dummy(dummy_user, dummy_user_bias):
    x = np.zeros(factors)
    y = np.zeros((factors, factors))
    for movie_id, rating in dummy_user:
        movie_index = movie_to_idx[movie_id]
        error = rating - dummy_user_bias - movies_biases[movie_index]
        x += movies_latent[:, movie_index] * error
        y += np.outer(movies_latent[:, movie_index].T,
                      movies_latent[:, movie_index])
    y += np.identity(factors) * tau
    return np.matmul(np.linalg.inv(lambd * y), lambd * x)

def remove_year(movie_title):
    return movie_title.split('(')[0]

def fetch_movie_id(movie_title):
    movie_title=remove_year(movie_title)
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_title
    }
    response = requests.get(TMDB_SEARCH_URL, params=params)
    if response.status_code == 200:
        results = response.json().get('results')
        if results:
            return results[0]['id']
    return None


def fetch_movie_details(movie_id):
    url = f"{TMDB_MOVIE_DETAILS_URL}{movie_id}"
    params = {'api_key': TMDB_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def get_best_match_movie_id(pattern, df):
    regex = re.compile(pattern, re.IGNORECASE)
    
    matches = df[df['title'].apply(lambda title: bool(regex.search(title)))]
    
    if not matches.empty:
        return matches.iloc[0]['movieId']  # Return the movieId of the best match
    else:
        return None  





@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", default="")
    search_pattern = user_id  
    movie_idx = get_best_match_movie_id(search_pattern, movies)
    print(movie_idx)
    if not movie_idx:
        return jsonify({"error": "Movie not found"}), 404


    rating = request.args.get("rating", default=5.0, type=float)

    user_dummy = [(movie_idx, rating)]
    dummy_user_latent = np.zeros(32)

    for _ in tqdm(range(30)):
        dummy_user_bias = calculate_dummy_user_bias(
            user_dummy, _, dummy_user_latent)
        dummy_user_latent = update_user_latent_dummy(
            user_dummy, dummy_user_bias)

    preds = [
        np.dot(dummy_user_latent.T,
               movies_latent[:, i]) + (0.05 * movies_biases[i])
        for i in range(len(idx_to_movie))
    ]

    pred_df = pd.DataFrame(preds, columns=["predictions"])
    pred_df = pred_df.sort_values("predictions", ascending=False)

    best_match = list(pred_df.head(5).index)
    recommendations = []

    for idx in best_match:
        movie_id = idx_to_movie[idx]
        movie_info = movies[movies["movieId"] == movie_id].iloc[0]
        tmdb_movie_id = fetch_movie_id(movie_info["title"])
        if tmdb_movie_id:
            details = fetch_movie_details(tmdb_movie_id)
            if details:
                recommendations.append({
                    "movieId": int(details["id"]),
                    "title": movie_info["title"],
                    "overview": details.get("overview", "No description available"),
                    "poster": f"{TMDB_IMAGE_BASE_URL}{details.get('poster_path')}" if details.get('poster_path') else "No Image Available",
                    "release_date": details.get("release_date", "N/A"),
                    "rating": details.get("vote_average", "N/A")
                })

    return jsonify(recommendations)


if __name__ == "__main__":
    app.run(debug=True)
