from utils.aws_funcs import upload_to_s3, download_from_s3, delete_folder_from_s3, get_from_s3
from projects.project_2.code_snippets import *
import pandas as pd
import numpy as np
import streamlit as st

def imports():
    with st.expander('Required Packeges: '):
        st.code(IMPORTS, language='python')

def display_code(code, text):
    with st.expander(text):
        st.code(code, language='python')

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

def get_csv(name):
    if file := get_from_s3(f'projects/Emoji Prediction/csv/{name}.csv'):
            if f'{name}_data_project_2' not in st.session_state:
                df = pd.read_csv(file['Body'])
                st.session_state[f'{name}_data_project_2'] = df 
            else:
                df = pd.read_csv(file['Body'])
    return df


def display_functions():
    menu =  ('Visualization', 'Preprocessing', 'Model Metrics', 'Amazon S3','Differents')
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

def base_model():
    st.markdown('''
        It is always a good practice to create a base model to understand how good or bad our predictions
        at least should be. As base model we'll use TfidfVectorizer for feature extraction and 
        Multinomial Naive Bayes classifier. 
    ''')
    display_code(MODEL_0, 'Model 0: Naive Bayes Classifier')

    st.markdown('We see that the minimum accuracy that we should achieve further is 44.2%')


def tensorflow_models():
    st.markdown(
        '''
            Now we are going to use tensorflow to create Deep Learning Models to beat the resul of the *Model 0*:

            1) Vectorize train and test sentences

            2) Create efficient tensorflow dataset from our vectorized data for both train and test sets (due function 
            that was introduced in earlier)

            3) Callbacks:

                - Dynamically reduce learning rate
                - Tried with Early Stopping Callback
        '''
    )


def embeddings():
    st.markdown('''
        In order to make our representation of words in tweets more useful it is the best practice to
        create an embedding of tokens of input words. It is implemented by TensorFlow Keras Embedding layer
    ''')
    display_code(EMBEDDINGS, 'Embedding Layer')


def model_1():
    st.markdown('''
        First model we'll try: ***LSTM***. 
    ''')
    display_code(MODEL_1, 'Model 1 Architecture')

def model_1_results():
    st.markdown('''
        It can be useful to take a look at Plot/Loss curves. It can help us to understand whether our model is able 
        to learn something more or not.
    ''')
    display_plot('model_1_plot_loss', 'Plot/Loss Curves')


def model_2():
    st.markdown('''
        Second model to try: ***Bidirectional***. 
    ''')
    display_code(MODEL_2, 'Model 2 Architecture')

def model_2_results():
    display_plot('model_2_plot_loss', 'Plot/Loss Curves')

def model_3():
    st.markdown('''
        Third model to try: *Con1D* with *MaxPooling1D* and *Bidirectional* layer after
    ''')
    display_code(MODEL_3, 'Model 3 Architecture')

def model_3_results():
    display_plot('model_3_plot_loss', 'Plot/Loss Curves')

def model_4():
    st.markdown('''
        Fourth Model is the most difficult, since we implement simple *Attention* algorithm
    ''')
    display_code(MODEL_4, 'Model 4 Architecture')

def model_4_results():
    display_plot('model_4_plot_loss', 'Plot/Loss Curves')

def loss_accuracy():
    display_code_csv(RESULTS, 'results', 'Metrics Table')

def confusion_matrices():
    st.markdown('''
        It is always useful to take a look at at the tables of results, where we can see where our model 
        making the errors the most, get confused.
    ''')
    with st.expander('Confusion Matrix'):
        for i in range(4):
            st.markdown(f'##### Model {i + 1}')
            st.write(get_csv( f'model_{i + 1}_confusion_matrix'))



def main():
    st.markdown('### Model 0: Bayes Naive')
    base_model()
   
    st.markdown('## Deep Learning Models')
    tensorflow_models()
    
    st.markdown('### Preprocessing')
    embeddings()

    st.markdown('### Model 1: *LSTM*')
    model_1()
    model_1_results()
   
    st.markdown('### Model 2: *Bidirectional*')
    model_2()
    model_2_results()

    st.markdown('### Model 3: *Conv1D + Bidirectional*')
    model_3()
    model_3_results()
    
    st.markdown('### Model 4: *LSTM with Attention*')
    model_4()
    model_4_results()

    st.markdown('## Results ')

    loss_accuracy()


    confusion_matrices()

    st.markdown('### Observations')
    with st.expander('Observations'):
        st.markdown('''
        We can see that the most diffuclt architecture is the most accurate one.
        Do not forget that our *Naive Bayes* Model, base one, gives the result of *44.2%*, which is wvwn higher than 
        *LSTM*. 

        Time of training the first three models is quite the same: *9-14* minutes, but we can see that in all 
        cases there is no point to train further than 8 - 10 epochs. The 4th model takes more time to train, since 
        *recurrent dropout* in the LSTM layer isn't supported by cuDNN. I experienced with it and without, results 
        differ about 1 percent and here with it presented. 
        ''')

        st.markdown('''
            If we talk about *Confusion Matrices*:

            1) Almost all categories in all models are confused with the most represented one - ***Sob***
            2) In general, the most represented categories are the most accurate
                
        ''')


    
if __name__ == "__main__":
    main()
