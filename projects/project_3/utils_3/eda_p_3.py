from utils.aws_funcs import upload_to_s3, download_from_s3, delete_folder_from_s3, get_from_s3
from projects.project_3.data.code_snippets.code_snippets import *
import pandas as pd
import numpy as np
import streamlit as st
from functools import partial

def get_from_s3_specific(file_name, format):
    return get_from_s3(f'projects/Bank_Marketing_Classification_Task/{format}/{file_name}.{format}')

def display_csv(name, text):
    try:
        if file := get_from_s3_specific(name, 'csv'):
            if f'{name}_data_project_3' not in st.session_state:
                df = pd.read_csv(file['Body'])
                st.session_state[f'{name}_data_project_3'] = df
        with st.expander(text):
            st.write(st.session_state[f'{name}_data_project_3'])
    except Exception:
            st.error('Object File is not Found')

def display_plot(name, text):
    try:
        if file := get_from_s3_specific(name, 'png'):
            with st.expander(text):
                st.image(file['Body'].read())
    except Exception:
            st.error('Object File is not Found')

def display_code_plot(name_code, name_plot, text):
    with st.expander(text):
        st.code(name_code, language='python')
        file = get_from_s3_specific(name_plot, 'png')
        st.image(file['Body'].read())

def display_code_csv(name_code, name_csv, text):
    with st.expander(text):
        st.code(name_code, language='python')
        file = get_from_s3_specific(name_csv, 'csv')
        df = pd.read_csv(file['Body'])
        st.write(df)


def imports():
    st.markdown('''
        First step is to import the required for the project packages:
    '''
    )

    with st.expander('Required Packeges: '):
        st.code(IMPORTS, language='python')

def display_functions():
    st.markdown('''
        Of course, it is necessary to create functions that help us to solve the problem. Let's take a look at them. 
    ''')
    menu =  ('Visualization', 'Statistical', 'Amazon S3')
    with st.expander('Functions For The Project'):
        menu_functions = st.radio('Choose the section of functions?', menu, index=None)
        if menu_functions == '':
            st.info('Choose the section of functions that You are interested in.')

        elif menu_functions == 'Visualization':
           st.code(VISUALIZATION) 

        elif menu_functions == 'Statistical':
            st.code(STATISTICAL)

        elif menu_functions == 'Amazon S3':
            st.markdown(
                '''
                This functions were used to upload Data(images, preprocessed df and so on) in all the stages to Amazon S3 bucket to retrieve it in any moment.
                '''
            )
            st.code(AMAZON_S3)


def raw_data():
    st.markdown('''
        Let's take a look at our raw data
    ''')

    display_code_csv(RAW_DATA, 'raw_data', 'Raw Data')

def without_null():
    st.markdown('''We can see immediately that there are the column "*Unnamed: 0*" that can be interpreted as duplicate of
        index column and it makes no sense to store this data in our DataFrame so we get rid of it
    ''')

    display_code_csv(WITHOUT_NULL, 'without_null', 'Data without excessive column')

def is_there_null():
    st.markdown('''
        We need to check if there are any NaNs values in our dataset, let's looks at this: 
    ''')

    display_csv('is_there_null', 'Check on Nulls')

def partition_numerical():
    st.markdown('''
        Now let's divide the DataSet into two parts: *numerical* and *categorical* to analyse is separately and take a look
        at *.describe* method to see the Statistics of each one
    ''')
    display_csv('numerical_df', 'Numerical Data')


def describe_numerical():
    display_csv('numerical_describe', 'Desription of Numerical Data')

def partition_categorical():
    display_csv('categorical_df', 'Categorical Data')

def describe_categorical():
    display_csv('categorical_describe', 'Desription of Categorical Data')

def distribution_age():
    display_plot('distribution_age', 'Age Distribution')

def distribution_balance():
    display_plot('distribution_balance', 'Balance Distribution')

def distribution_campaign():
    display_plot('distribution_campaign', 'Campaign Distribution')

def distribution_day():
    display_plot('distribution_day', 'Day Distribution')

def distribution_duration():
    display_plot('distribution_duration', 'Duration Distribution')

def distribution_pdays():
    display_plot('distribution_pdays', 'PDays Distribution')

def distribution_previous():
    display_plot('distribution_previous', 'Previous Distribution')

def correlation_matrix():
    display_code_plot(CORRELATION_MATRIX, 'correlation_matrix', 'Correlation Matrix')

def multicollinearity():
    display_code_csv(MULTICOLLINEARITY, 'vif_data', 'VIF Coefficienet')
    st.markdown("We see no coefficient exceeds 6-10, so we can assume that there is no such a phenomenon" )


def distribution_job():
    display_plot('distribution_job', 'Job Distribution')

def distribution_marital():
    display_plot('distribution_marital', 'Marital Distribution')

def distribution_education():
    display_plot('distribution_education', 'Education Distribution')

def distribution_default():
    display_plot('distribution_default', 'Default Distribution')

def distribution_housing():
    display_plot('distribution_housing', 'Housing Distribution')

def distribution_loan():
    display_plot('distribution_loan', 'Loan Distribution')

def distribution_contact():
    display_plot('distribution_contact', 'Contact Distribution')

def distribution_month():
    display_plot('distribution_month', 'Month Distribution')

def distribution_poutcome():
    display_plot('distribution_poutcome', 'Poutcome Distribution')

def chi2_test():
    st.markdown('''
        We want to find how our categorical features are related to the target variable and
        so we conduct Chi2 Test.
    ''')
    display_code_csv(CHI2_TEST, 'chi2_test', 'Results of Chi2 Testing')
    st.markdown('''We see that only one feature "*default*" exceeds significance level of $p = 0.05$, the 
    others one largest value is 0.001.
    ''')



    




def main():
    
    st.markdown('#### First Steps')
    
    imports()

    
    display_functions()

    st.markdown('#### First Time with Data')

    raw_data()

    without_null()    

    is_there_null()

    partition_numerical()

    describe_numerical()

    partition_categorical()

    describe_categorical()

    st.markdown('#### EDA')
    
    st.markdown('#### Numerical Features')
    st.markdown('##### Distribution of Numerical Features')

    distribution_age()
    distribution_balance()
    distribution_campaign()
    distribution_day()
    distribution_duration()
    distribution_pdays()
    distribution_previous()

    st.markdown('##### Correlation Matrix')
    correlation_matrix()

    st.markdown('##### Multicollinearity')
    multicollinearity()

    st.markdown('#### Categorical Features')
    st.markdown('##### Distribution of Categorical Features')

    distribution_job()
    distribution_marital()
    distribution_education()
    distribution_default()
    distribution_housing()
    distribution_loan()
    distribution_contact()
    distribution_month()
    distribution_poutcome()

    st.markdown('##### Chi2 Test')
    chi2_test()
    















    



if __name__ == "__main__":
    main()
