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

def display_csv_general(name, text):
    try:
        if file := get_from_s3(f'projects/Stock_Price_analysis/csv/{name}.csv'):
            if f'{name}_data_project_1' not in st.session_state:
                df = pd.read_csv(file['Body'],)
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
        For this purpose we also use the function that was declared in the first section - *plot_loss_curves*
    ''')

    display_plot('3_plot_loss_model_1', 'Plot Loss Curves of Model 1')

    st.markdown('It is obvious from the plot, that number of epochs is excessive. In general, we can reduce it')


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


def model_2():
    st.markdown('''
        Now we will train our second model.
    '''
    )

    display_code(MODEL_2, 'Model 2 Structure')

def model_2_plot_loss():
    st.markdown('''
        Now we can take a look at plot loss curves to estimate whether we have chances to learn something more.
        For this purpose we also use the function that was declared in the first section - *plot_loss_curves*
    ''')

    display_plot('plot_loss_model_2', 'Plot Loss Curves of Model 2')

    st.markdown('It is obvious from the plot, that number of epochs is excessive. In general, we can reduce it')


def model_2_results():
    # st.markdown(
    #     '''

    #     '''
    # )
    pass

def model_2_results_plot():
    st.markdown(
        '''
        Let's take a look at the plot that shows comparison between real prices and predicted ones
        '''
    )
    display_plot('preds_true_model_2', 'Model 2 Results')


def model_3():
    display_code(MODEL_3, 'Models 3 Parameters')

def model_3_training():
    display_code(MODEL_3_TRAINING, 'Model 3 Training')


def model_3_results():
    display_csv_general('res_ML_without_features', 'Model 3 Results')

def model_3_feature_importance():
    st.markdown(
        '''
        Since we used *XGBoost* it is appropriate to look at the feature importance
        '''
    )
    display_code_plot(FEATURE_IMPORTANCE_WITHOUT, 'xgboost_feature_importance_without', 'Feature Importance')

    st.markdown('From the plot we can see that the most close *Close* price to current day logically is the most influencial')


def feature_addition():
    st.markdown('We want to add some features that can make our results even more accurate')
    st.markdown('''
    
    - Volume shocks - If volume traded is 10% higher/lower than the previous day - make a 0/1 boolean time series for shock, 0/1 dummy-coded time series for the direction of shock. 
    
    - Price shocks - If the closing price at T vs T+1 has a difference > 2%, then 0/1 boolean time series for shock, 0/1 dummy-coded time series for the direction of shock. 
    
    - Pricing black swan - If the closing price at T vs T+1 has a difference > 2%, then 0/1 boolean time series for shock, 0/1 dummy-coded time series for the direction of shock. 
    
    - Pricing shock without volume shock - based on points 3.1 & 3.2 - Make a 0/1 dummy time series.
                
''')
    
def new_data_show():
    display_code_csv(NEW_DF_WITH_FEATURES, 'window_7_with_features', 'Data with new Features')



def model_4():
    st.markdown('''
        Now we can try our new featured data.
    '''
    )

    display_code(MODEL_4, 'Model 4')

def model_4_plot_loss():
    st.markdown('''
        Now we can take a look at plot loss curves to estimate whether we have chances to learn something more.
        For this purpose we also use the function that was declared in the first section - *plot_loss_curves*
    ''')

    display_plot('plot_loss_model_4', 'Plot Loss Curves of Model 4')

    st.markdown('Here we can see that stil there is place for improvement, losses and MAE are smoothly going down ')


def model_1_results():
    # st.markdown(
    #     '''

    #     '''
    # )
    pass

def model_4_results_plot():
    st.markdown(
        '''
        Let's take a look at the plot that shows comparison between real prices and predicted ones
        '''
    )
    display_plot('preds_true_model_4', 'Model 4 Results')


def model_5():
    display_code(MODEL_5, 'Models 5 Parameters')

def model_5_training():
    display_code(MODEL_5_TRAINING, 'Model 5 Training')


def model_5_results():
    display_csv_general('res_ML_with_features', 'Model 5 Results')

def model_5_feature_importance():
    st.markdown(
        '''
        And again, since we used *XGBoost* it is appropriate to look at the feature importance.
        '''
    )
    display_code_plot(FEATURE_IMPORTANCE_WITH, 'xgboost_feature_importance_with', 'Feature Importance')

    st.markdown('''From the plot we can see that the most influencial now is also the closest to current day price,
                but also *Volume* plays very important role in predictions
                ''')



def main():
    st.markdown('#### Introduction')
    title()

    st.markdown('#### Preprocessing')
    preprocessing()

    st.markdown('### Modeling')

    st.markdown('#### Model 1: Dense')

    model_1()

    model_1_plot_loss()

    model_1_results()

    model_1_results_plot()



    st.markdown('#### Model 2: Conv1D')

    model_2()

    model_2_plot_loss()

    model_2_results()

    model_2_results_plot()

    st.markdown('#### Block of Models 1')

    st.markdown('Not only Deep Learning models are useful is predicting, we try now ML algorithms')

    model_3()

    model_3_training()

    model_3_results()

    model_3_feature_importance()


    st.markdown('### Add Features')
    feature_addition()
    new_data_show()

    st.markdown('#### Model 4: Dense')

    model_4()

    model_4_plot_loss()

    model_4_results()

    model_4_results_plot()

    st.markdown('#### Block of Models 2')

    model_5()

    model_5_training()

    model_5_results()

    model_5_feature_importance()

    

if __name__ == "__main__":
    main()