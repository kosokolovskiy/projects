import streamlit as st
import requests

def get_total_movies():
    url = 'http://127.0.0.1:5000/get-total-movies'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        total_movies = data.get('total')
        if total_movies is not None:
            return total_movies
    else:
        st.error("Failed to retrieve the total number of movies. Please ensure the Flask API is running and accessible.")

def select_user_id():
    return st.selectbox('Choose User Id: ', list(range(1, 307)), index=None)

def get_movie_id_from_title(title):
    url = f'http://127.0.0.1:5000/get-movie-id?title={title}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('movie_id')
    else:
        st.error("Movie not found. Please check the title and try again.")
        return None

def select_movie_title():
    url = 'http://127.0.0.1:5000/get-movie-titles'
    response = requests.get(url)
    if response.status_code == 200:
        movie_titles = response.json()
        return st.selectbox('Choose Movie Title: ', movie_titles, index=None)
    else:
        st.error("Failed to retrieve movie titles. Please ensure the Flask API is running and accessible.")
        return None

def check_rated(user_id, movie_id):
    url = f'http://127.0.0.1:5000/get-rating?user_id={user_id}&movie_id={movie_id}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        rating = data.get('rating')
        if rating is not None:
            st.info(f"You have already rated this movie with the rate: {rating}")
            return True
    return False

def give_rating(user_id, movie_id):
    rating = st.number_input('Choose the rating', min_value=0.5, max_value=5.0, step=0.5, value=4.0)
    if st.button('Submit', key='submit_rating'):
        st.success(f'Success, you rated the movie with: {rating}')


def rate_movie_main():
    st.title("Rate a Movie")
    st.markdown('''
        ***Instructions:*** 
        1. Please choose the "*User Id*" that corresponds to yours.
        2. Choose the name of the film you want to give a rating to.
        3. Please choose the rating you want to give to the chosen movie and click "*Submit*" button.
    ''')

    user_id = select_user_id()
    movie_title = select_movie_title()
    if movie_title:
        movie_id = get_movie_id_from_title(movie_title)
        if movie_id and not check_rated(user_id, movie_id):
            give_rating(user_id, movie_id)

if __name__ == '__main__':
    rate_movie_main()
