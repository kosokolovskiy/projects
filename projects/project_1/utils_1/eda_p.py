from utils.aws_funcs import upload_to_s3, download_from_s3, delete_folder_from_s3, get_from_s3
from projects.project_1.code_snippets import *
import pandas as pd
import numpy as np
import streamlit as st

def imports():
    with st.expander('Required Packeges: '):
        st.code(IMPORTS, language='python')
    

def display_csv(name, text):
    try:
        if file := get_from_s3(f'projects/Stock_Price_analysis/csv/{name}.csv'):
            if f'{name}_data_project_1' not in st.session_state:
                df = pd.read_csv(file['Body'], parse_dates=['Date'], index_col='Date')
                st.session_state[f'{name}_data_project_1'] = df
        with st.expander(text):
            st.write(st.session_state[f'{name}_data_project_1'])

    except Exception:
            st.error('Object File is not Found')


def display_plot(name, text):
    try:
        print(f'projects/Stock_Price_analysis/plots/{name}.png')
        if file := get_from_s3(f'projects/Stock_Price_analysis/plots/{name}.png'):
            with st.expander(text):
                st.image(file['Body'].read())
    except Exception:
            st.error('Object File is not Found')


def display_code_plot(name_code, name_plot, text):
    with st.expander(text):
        st.code(name_code, language='python')
        file = get_from_s3(f'projects/Stock_Price_analysis/plots/{name_plot}.png')
        st.image(file['Body'].read())

def display_functions():
    menu =  ('Visualization', 'Predictions', 'Sliding Window', 'Amazon S3','Differents')
    with st.expander('Functions For The Project'):
        menu_functions = st.radio('Choose the section of functions?', menu, index=None)
        if menu_functions == '':
            st.info('Choose the section of functions that You are interested in.')

        elif menu_functions == 'Visualization':
           st.code(VISUALIZATION) 

        elif menu_functions == 'Predictions':
            st.code(PREDICTIONS)

        elif menu_functions == 'Sliding Window':
            st.markdown('''
                There were two approaches used here. 

                Firstly, I used Numpy Functions, but further changed to Pandas and in the folloing project Pandas function is used.
                
                I see it as more user-friendly, although for a bigger dataset numpy is more efficient.
                ''')
            st.code(SLIDING_WINDOW)
        elif menu_functions == 'Differents':
            st.code(DIFFERENTS)

        elif menu_functions == 'Amazon S3':
            st.markdown(
                '''
                This functions were used to upload Data(images, preprocessed df and so on) in all the stages to Amazon S3 bucket to retrieve it in any moment.
                '''
            )
            st.code(AMAZON_S3)

def raw_data():
    display_csv('raw', text='Raw Data')

def first_preprocess():
    display_csv('cleaned', 'Final Version')

def plot_1():
    display_plot('1_overview', 'Overview')

def train_test_split():
    with st.expander('Train Test Split'):
        st.code(TRAIN_TEST_SPLIT)
    display_plot('2_train_test_split', text='Train Test Split')




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

    st.markdown('''
        As the goal of Project suggests, we need only the *'Close Price'*. So firstly, we read a *.csv* file, parsing dates and
        making it as an index column. Secondly, we get rid of all the columns except *'Close'* and *'Volume'*. Now we can look at
        final Dataframe that we'll work with and compare with the Raw one.
    ''')
    raw_data()
    first_preprocess()

    st.markdown('''
    Now we can take a look at gpraphical representation of the information given to us by dataframe.
    ''')
    plot_1()

    st.markdown('''
        Now we'll do Train/Test split with test size of 20\% and take a look what are we're trying to predict. 
        '''
    )
    display_code_plot(TRAIN_TEST_SPLIT, '2_train_test_split', 'Train/Test Split')




if __name__ == "__main__":
    main()
