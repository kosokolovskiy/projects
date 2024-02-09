import streamlit as st

def main():
    st.markdown('''
        ## Predicting Emojis in Tweets

        ##### Assigment

        The task is to build a predictive model of emoji for a given piece of text - tweet.


        ##### Data Description

        **Two files:**

        - *tweets.txt*, where each line is the text of a tweet with corresponding emoji;
        - *emoji.txt*, where each line is the name of the emoji for the corresponding text in *tweets.txt*.
    ''')

if __name__ == '__main__':
    main()