import streamlit as st

def select_user_id():  # sourcery skip: inline-immediately-returned-variable
    user_id = st.selectbox('Choose User Id: ', list(range(1, 100)), index=None)
    return user_id

def select_movie_id():  # sourcery skip: inline-immediately-returned-variable
    movie_id = st.selectbox('Choose Movie Id: ', list(range(1, 100)), index=None)
    return movie_id

def show_recommended_movie():
    st.markdown('Our Recommendations to You: ')

def get_recommendation_main():
    st.markdown('''
        ***Instruction:*** 

        1) Please choose the "*User Id*" that corresponds yours

        2) Choose the name of film you want to get recommendation to

        3) Choose one of the recommended movies and enjoy!

    ''')

    col1, col2 = st.columns(2)

    with col1:
        user_id = select_user_id()
        st.session_state['user_id'] = user_id

    with col2:
        movie_id = select_movie_id()
        st.session_state['movie_id'] = movie_id

    if st.session_state['user_id'] and st.session_state['movie_id']:
        with st.expander('Movies For You: ', expanded=True):
            show_recommended_movie()


if __name__ == '__main__':
    get_recommendation_main()