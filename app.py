import pandas as pd
import pickle
import streamlit as st
import numpy as np
movies_list_df=pickle.load(open('movies.pkl','rb'))

movies_list1=movies_list_df['title'].values
st.title('Movie Recommender System')
selected_movie_name=st.selectbox("Select any movie to get a recommendation of 5 movies.",movies_list1)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_list_df['tags']).toarray()
similarity = cosine_similarity(vectors)
def recommend(movie):
    movie_index =movies_list_df[movies_list_df['title'] == movie].index[0]  # 1st matching movie to prevent duplicates
    distances = similarity[movie_index]
    movies_list_ = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]#it has all 5 movie indexes
    recommended_movies = []
    for i in movies_list_:
        movie_id=i[0]
        recommended_movies.append(movies_list_df.iloc[i[0]].title)
    return recommended_movies
if st.button('Recommend'):
    for i in recommend(selected_movie_name):
        st.write(i)