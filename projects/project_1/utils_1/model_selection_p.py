from utils.aws_funcs import upload_to_s3, download_from_s3, delete_folder_from_s3, get_from_s3
from projects.project_1.code_snippets import *
import pandas as pd
import numpy as np
import streamlit as st

def display_csv(name, text, *args):
    try:
        if file := get_from_s3(f'projects/Stock_Price_analysis/csv/{name}.csv'):
            if f'{name}_data_project_1' not in st.session_state:
                df = pd.read_csv(file['Body'], parse_dates=['Date'], index_col='Date')
                st.session_state[f'{name}_data_project_1'] = df
        with st.expander(text):
            st.write(st.session_state[f'{name}_data_project_1'])
    except Exception:
            st.error('Object File is not Found')


def display_plot(name, text, *args):
    try:
        print(f'projects/Stock_Price_analysis/plots/{name}.png')
        if file := get_from_s3(f'projects/Stock_Price_analysis/plots/{name}.png'):
            with st.expander(text):
                st.image(file['Body'].read())
    except Exception:
            st.error('Object File is not Found')

def display_code(text_code, text, *args):
    with st.expander(text):
        st.code(text_code, language='python')

def display_code_plot(name_code, name_plot, text, *args):
    with st.expander(text):
        st.code(name_code, language='python')
        file = get_from_s3(f'projects/Stock_Price_analysis/plots/{name_plot}.png')
        st.image(file['Body'].read())


def display_code_csv(name_code, name_csv, text, *args):
    with st.expander(text):
        st.code(name_code, language='python')
        file = get_from_s3(f'projects/Stock_Price_analysis/csv/{name_csv}.csv')
        df = pd.read_csv(file['Body'], parse_dates=['Date'], index_col='Date')
        st.write(df)
        if args:
            for elem in args:
                file = get_from_s3(f'projects/Stock_Price_analysis/csv/{elem}.csv')
                df = pd.read_csv(file['Body'], parse_dates=['Date'], index_col='Date')
                st.write(df)


def title():
    st.markdown('''

        **Now we will work with several types of models and compare the results of learning:**
            
        1) Model 1: Dense

        2) Model 2: Conv1D

        3) Block of Models:
            - Ridge;
            - Lasso;
            - XGBoost;

        4) Model 3: Dense with Additional Features

        5) Block of Models with additional features:
            - Ridge;
            - Lasso;
            - XGBoost;
    '''
    )

def preprocessing():
    st.markdown('''
        First we need to prepare our data for forecasting. We generate new features by shifting specified columns
        backward according to window size and horizon, effectively creating a lagged features. The functions were introduced in the
        first section and can be seen there.
    ''')

    display_code_csv(MAKE_WINDOWS_LABELS, 'window_7_without_features', 'Windowed Features and Labels', 'label_without_features')

def model_1():
    st.markdown('''
        Finally, we can create our first simple model to try to predict *Close* price of the next day.

        It is used EarlyStopping callback with parameters that can be observed lower
    '''
    )

    display_code(MODEL_1, 'Model 1 Structure')

def model_1_plot_loss():
    st.markdown('''
        Now we can take a look at plot loss curves to estimate whether we have chances to learn something more.
        For this purpose we also use the function what was declared in the first section - *plot_loss_curves*
    ''')

    display_plot('3_plot_loss_model_1', 'Plot Loss Curves of Model 1')


def model_1_results():
    # st.markdown(
    #     '''

    #     '''
    # )
    pass

def model_1_results_plot():
    st.markdown(
        '''
        Let's take a look at the plot that shows comparison between real prices and predicted ones
        '''
    )
    display_plot('4_preds_true_model_1', 'Model 1 Results')

# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )

# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )


# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )


# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )


# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )


# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )


# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )


# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )


# def model_1_results():
#     st.markdown(
#         '''
        
#         '''
#     )





def main():
    title()

    preprocessing()

    model_1()

    model_1_plot_loss()

    model_1_results()

    model_1_results_plot()


if __name__ == "__main__":
    main()