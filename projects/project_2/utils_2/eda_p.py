from utils.aws_funcs import upload_to_s3, download_from_s3, delete_folder_from_s3, get_from_s3
from projects.project_2.code_snippets import *
import pandas as pd
import numpy as np
import streamlit as st

def imports():
    with st.expander('Required Packeges: '):
        st.code(IMPORTS, language='python')
    

def display_csv(name, text):
    try:
        if file := get_from_s3(f'projects/Emoji Prediction/csv/{name}.csv'):
            if f'{name}_data_project_1' not in st.session_state:
                df = pd.read_csv(file['Body'])
                st.session_state[f'{name}_data_project_1'] = df
        with st.expander(text):
            st.write(st.session_state[f'{name}_data_project_1'])
    except Exception:
            st.error('Object File is not Found')


def display_plot(name, text):
    try:
        if file := get_from_s3(f'projects/Emoji Prediction/plots/{name}.png'):
            with st.expander(text):
                st.image(file['Body'].read())
    except Exception:
            st.error('Object File is not Found')


def display_code_plot(name_code, name_plot, text):
    with st.expander(text):
        st.code(name_code, language='python')
        file = get_from_s3(f'projects/Emoji Prediction/plots/{name_plot}.png')
        st.image(file['Body'].read())

def display_code_csv(name_code, name_csv, text):
    with st.expander(text):
        st.code(name_code, language='python')
        file = get_from_s3(f'projects/Emoji Prediction/csv/{name_csv}.csv')
        df = pd.read_csv(file['Body'])
        st.write(df)

def display_functions():
    menu =  ('Visualization', 'Preprocessing', 'Model Metrics', 'Results', 'Amazon S3', 'Differents')
    with st.expander('Functions For The Project'):
        menu_functions = st.radio('Choose the section of functions?', menu, index=None)
        if menu_functions == '':
            st.info('Choose the section of functions that You are interested in.')

        elif menu_functions == 'Visualization':
           st.code(VISUALIZATION) 

        elif menu_functions == 'Preprocessing':
            st.code(PREPROCESSING)

        elif menu_functions == 'Model Metrics':
            st.code(MODEL_METRICS)

        elif menu_functions == 'Amazon S3':
            st.markdown(
                '''
                This functions were used to upload Data(images, preprocessed df and so on) in all the stages to Amazon S3 bucket to retrieve it in any moment.
                '''
            )
            st.code(AMAZON_S3)

        elif menu_functions == 'Results':
            st.code(RESULT_FUNC)

def raw_data():
    st.markdown('''
        Initially we have two files one with tweets and the second with corresponding emoji. Let's take
        a look at them
    ''')
    display_csv('raw_emoji', text='Raw Emoji')
    display_csv('raw_tweets', text='Raw Tweets')



def distribution_labels():
    st.markdown('''
        Now it would be useful to look at distribution of labels
    ''')
    display_code_plot(DISTRIBUTION_LABELS, '1_categories_count', 'Distribution of Labels')

def max_to_min_ratio():
    st.markdown('''
        It would be possibly useful to see closer how unbalanced our data, so let's plot
        how the most popular category, *Sob*, relates to all others
    ''')
    
    display_code_plot(MAX_MIN_RATION, '2_unbalanced_check', 'Ratio of Max to Others')

    st.markdown('''
        The max ratio is about 5 to *relaxed* and *flushed*, so maybe some balancing technics will be 
        as a bonus for good predictions
    ''')

def distribution_tweet_length():
    st.markdown('''
        Now let's take a look at distribution of Tweet Lengths
    ''')
    display_code_plot(TWEETS_LENGTH_DISTRIBUTION, 'Tweets_Length_Distribution', 'Tweets Length Distribution')


def preprocess_tweets():
    st.markdown('''
        Now it is useful to preprocess our raw text of tweets into more interpretable outlook:

        1)  removes all non-alphanumeric characters from the text. 

            Only letters (a-z, A-Z) and numbers (0-9) are left

        2) strip multiple spaces;

        3) strip punctuation;

        4) stemming words;

        5) remove stop words 

            Helps to focus on meaningful words.
    ''')

    display_code_csv(PREPROCESS_TWEETS, 'preprocessed_tweets', 'Preprocessed Tweets')


def distribution_tweet_length_symbols():
    st.markdown('''
        Now, it can be interesting to look at how and is really distribution of tweets in symbols changed
    ''')

    display_code_plot(TWEETS_LENGTH_SYMBOLS_DISTRIBUTION, 'Tweet_Length_Symbols_Distribution', 'Tweet Length in Symbols Distribution')


def distribution_tweet_length_symbols():
    st.markdown('''
        Now, it can be interesting to look at  distribution of tweets in symbols 
    ''')

    display_code_plot(TWEETS_LENGTH_SYMBOLS_DISTRIBUTION, 'Tweet_Length_Symbols_Distribution', 'Tweet Length in Symbols Distribution')

def distribution_tweet_length_words():
    st.markdown('''
        Despite the number of symbols, somehow it can be beneficial to know the distribution of number of words
        in our tweets.
    ''')
    display_code_plot(TWEETS_LENGTH_WORDS_DISTRIBUTION, 'Tweet_Length_Words_Distribution', 'Tweet Length in Words Distribution')

def label_encoder():
    st.markdown(
        '''
            Now the time to encode our targets. It is implemented by sklearn *OrdinalEncoder*
        '''
    )
    display_code_csv(ENCODER, 'encoding', 'Encoding Table')

def train_test_split():
    st.markdown('''
        Now we're ready to combine numerical labels and preprocessed text of tweets in one dataframe and 
        to make train/test split with 20% as test size
    ''')

    display_code_csv(TRAIN_TEST_SPLIT, 'head_final_df', text='Train Test Split')


def convert_to_numbers():
    st.markdown('''
        Since our computer is not able to understand text in a form we do, it is required to trnsform 
        it into numbers.
        We'll use the dictionary of maximum number of words as 10_000 with fixed output length of 15.
    ''')

    with st.expander('Tokenizer'):
        st.code(TEXT_VECTORIZING)
        st.code(VOCABULARY)






def main():
    st.markdown('''
        First step is to import the required for the project packages:
    '''
    )
    imports()

    st.markdown('''
        Of course, it is necessary to create functions that help us to solve the problem. Let's take a look at them. 
    ''')
    display_functions()

    raw_data()

    distribution_labels()

    max_to_min_ratio()

    distribution_tweet_length()

    preprocess_tweets()

    distribution_tweet_length_symbols()

    distribution_tweet_length_words()

    label_encoder()

    train_test_split()

    convert_to_numbers()

if __name__ == "__main__":
    main()
