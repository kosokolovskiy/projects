import streamlit as st
from projects.project_5.utils.get_recommendation import get_recommendation_main
from projects.project_5.utils.rate_movie import rate_movie_main

def main_widget():

    options = ['Get Recommendations', 'Rate the Movie']
    choice = st.selectbox('What do you want to do?', options, index=None, key='options_to_choose')

    if choice == options[0]:
        try:
            get_recommendation_main()
        except Exception:
            st.info('This Feature is not supported yet')
    elif choice == options[1]:
        try:
            rate_movie_main()
        except Exception:
            st.info('This Feature is not supported yet')



if __name__ == '__main__':
    main_widget()