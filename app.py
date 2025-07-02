import pandas as pd
import pickle
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your pickled movie data
movies_list_df = pickle.load(open('movies.pkl', 'rb'))

# Create CountVectorizer and compute similarity matrix from 'tags' column
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_list_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Movie list for dropdown
movies_list1 = movies_list_df['title'].values

# Streamlit UI
st.title('Movie Recommender System')
selected_movie_name = st.selectbox("Select any movie to get a recommendation of 5 movies.", movies_list1)

# Recommendation logic
def recommend(movie):
    movie_index = movies_list_df[movies_list_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list_ = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    for i in movies_list_:
        recommended_movies.append(movies_list_df.iloc[i[0]].title)
    return recommended_movies

# Button to trigger recommendations
if st.button('Recommend'):
    for i in recommend(selected_movie_name):
        st.write(i)
