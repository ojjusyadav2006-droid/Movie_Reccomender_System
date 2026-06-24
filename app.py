import os
import pickle
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

if not os.path.exists('similarity.pkl'):
    movies_temp = pickle.load(open('movie.pkl', 'rb'))
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_temp['tags']).toarray()
    similarity_temp = cosine_similarity(vectors)
    pickle.dump(similarity_temp, open('similarity.pkl', 'wb'))

movies = pickle.load(open('movie.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

@st.cache_data
def fetch_poster(movie_id):
    for attempt in range(5):
        try:
            response = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=d209499ba124b4382f3acc728beee75e",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            data = response.json()
            if 'poster_path' in data and data['poster_path']:
                return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        except Exception:
            continue
    return "https://placehold.co/500x750?text=No+Poster"

@st.cache_data
def fetch_imdb_rating(title):
    for attempt in range(5):
        try:
            response = requests.get(
                f"http://www.omdbapi.com/?t={title}&apikey=4c4d4caa",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            data = response.json()
            return data.get('imdbRating', 'N/A')
        except Exception:
            continue
    return "N/A"

def fetch_data(movie_id, title):
    poster = fetch_poster(movie_id)
    rating = fetch_imdb_rating(title)
    return poster, rating

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    movie_ids = []
    titles = []
    for i in distances[1:6]:
        recommended_movies.append(movies.iloc[i[0]].title)
        movie_ids.append(movies.iloc[i[0]].movie_id)
        titles.append(movies.iloc[i[0]].title)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(fetch_data, movie_ids, titles))

    recommended_movies_posters = [r[0] for r in results]
    recommended_movies_ratings = [r[1] for r in results]
    return recommended_movies, recommended_movies_posters, recommended_movies_ratings

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title('Movie Recommender System')
st.caption("Pick a movie you like and get 5 similar picks")

selected_movie_name = st.selectbox(
    "Select your favorite movie",
    movies['title'].values,
)

if st.button('Recommend'):
    with st.spinner('Fetching recommendations...'):
        names, posters, ratings = recommend(selected_movie_name)
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.image(posters[idx], width='stretch')
            st.markdown(f"<p style='text-align: center; font-weight: 500;'>{names[idx]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: gold;'>⭐ {ratings[idx]} IMDB</p>", unsafe_allow_html=True)