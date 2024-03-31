import streamlit as st
from utils.get_recommendation import get_recommendation_main
from utils.rate_movie import rate_movie_main

def main_widget():

    options = ['Get Recommendations', 'Rate the Movie']
    choice = st.selectbox('What do you want to do?', options, index=None, key='options_to_choose')

    if choice == options[0]:
        get_recommendation_main()
    elif choice == options[1]:
        rate_movie_main()



if __name__ == '__main__':
    main_widget()