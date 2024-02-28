from utils.aws_funcs import get_from_s3
from projects.project_3.data.code_snippets.code_snippets import *
import pandas as pd
import numpy as np
import streamlit as st


def display_code(code, text):
    with st.expander(text):
        st.code(code, language='python')


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

def base_train_test_data_preparation():
    st.markdown("Let's split the data into train and test parts")
    display_code(BASE_TRAIN_TEST_DATA_PREPARATION, 'Train / Test Split for the Base Model')

def base_models():
    st.markdown("As the base models we're using the following models with default parameters:")
    display_code(BASE_MODELS, 'Base Models Settings')

def c_v_object():
    st.markdown('In order to get less biased estimation we create StratifiedKFold')
    display_code(C_V_OBJECT, 'StratifiedKFold')

def base_results():
    st.markdown('''
    Now we're training our model due to *train_models* function and
    observe the following results:
    ''')
    display_csv('base_results_train', 'Base Model Train Results')
    display_csv('base_results_test', 'Base Model Test Results')

    st.markdown('''
        We can see that the accuracy is pretty the same across all the models.
        However SVC cannot give any significant results so we get rid of this model
        in the next discussions. 

        The most important metric here for us - ***Precision***. The best result 
        in this category at the moment is shown by XGBoost, but the difference between results
        can be due to some parametric choices. So it would be better not to 
        get rid of the first and second models at this moment of time.
        Later we'll try to optimize the parameters of these models and improve metrics,
        but for now let's look at feature importance that *XGBoost* introduces to us.
    ''')
    

def base_feature_importance():
    display_plot('base_feature_importance', 'Feature Importance by XGBoost')
    st.markdown('''
        We can observe th difference importances of features in prediction
    ''')

def base_confusion_matrix():
    st.markdown('It is always useful to look at confusion matrix')
    with st.expander('Base Confusion Matrix'):
        names = [
            'base_Logistic Regregression',
            'base_Random Forest',
            'base_XGBoost',
            'base_SVC'
        ]
        for name in names:
            file = get_from_s3_specific(name, 'png')
            st.image(file['Body'].read())
            


def base_top_features():
    st.markdown('Replot the feature importance plot in different view to easily choose the number of features')
    display_code_plot(BASE_TOP_FEATURE_IMPORTANCE, 'base_feature_importance_1', 'Base Model Top Features Selection Plot')


def base_top_features_results():
    display_csv('base_results_top_features_train', 'Base Top Features Model Train Results')
    display_csv('base_results_top_features_test', 'Base Top Features Model Test Results')

    st.markdown('As it was guessed, performance of predictions is not changed')


def grid_search():
    st.markdown('''
        In this subsection we will try to find best parameters for our models resorting to *Grid Search*
    ''')
    display_code(BASE_GRID_SEARCH, 'Grid Search with Paramaters')



def grid_search_params():
    st.markdown("Now take a look at the best parameters that our exhaustive search was able to find")
    display_csv('base_best_params', 'Base Model Best Parameters')


def base_best_models_train():
    st.markdown('Now we train our models with the best parameters and look at the results')

    with st.expander('Results of Models with best Parameters'):
        st.code(BASE_BEST_MODELS_TRAIN, language='python')
        names = ['base_models_best_results_train', 'base_models_best_results_test']
        for name in names:
            file = get_from_s3_specific(name, 'csv')
            df = pd.read_csv(file['Body'])
            st.write(df)

    st.markdown('''We can observe significant improvements in final results in *XGBoost* model
        in both Train and Test datasets.
        However, it would be interesting whether we are able to improve it even better due to *Feature Engineering*
     ''')

def age_engineering():
    st.markdown('''We have seen in *EDA* that *Age* shows rigth-skewed distribution. To fix it
    we apply *log* transformation on the whole distribution.
    ''')
    display_plot('distribution_age_engineered', 'Age Logged Distribution')

def balance_engineering():
    st.markdown('''
        To fix *Balance* distribtuion, which is also right-skewed, we cannot simply apply *log transformation* since
        there are negative values, which are about 10\% of the whole dataset. Also there are some outliers that can be
        dropped from dataset, since there are only 4 values of balance over than 30 000. To handle negative value for
        the following *log transformation* we add the absolute minimum of *Balance* feature to all values and apply 
        *log*. It is still not the best distribution one desires to obtain, but it is better.
    ''')

    display_code_plot(BALANCE_ENGINEERED, 'distribution_balance_engineered', 'Balance Logged Distribution')


def duration_engineering():
    st.markdown('''In this case we apply *log transformation* directly, but with getting rid of outliers that are
    larger than 2000.
    ''')
    display_plot('distribution_balance_engineered', 'Duration Logged Distribution')


def campaign_engineering():
    st.markdown('''In this case we will apply different to the previous kind of featuring: we separate data
        into 4 categories. And also calculate chi2 statistics to find *p-value*, which will be $2.268e-05$
    ''')

    display_code(CAMPAIGN_ENGINEERED, 'Campaign Engineering')

def pdays_engineering():
    st.markdown('''By this numerical value we create 3 categories and also calculate the dependence between 
    *target* variable and feature.
    ''')
    display_code(PDAYS_ENGINEERED, 'Pdays Engineering')
    
def previous_engineering():
    display_code(previous_engineered, 'Previous Engineering')





def main():
    st.markdown('### Base Model. No Feature Engineering')

    base_train_test_data_preparation()

    base_models()

    c_v_object()

    base_results()

    base_feature_importance()

    base_confusion_matrix()

    st.markdown('#### Top Features for Base Model')

    st.markdown('''In the previous subsection we saw that not every feature is of the most importance, so now 
    we are choosing the N-top features and retrain model to prove that performance of the model is not reduced drastically
    ''')

    base_top_features()

    base_top_features_results()

    st.markdown('#### Grid Search Optimization of Base Models')

    grid_search()

    grid_search_params()

    base_best_models_train()

    st.markdown('### Feature Engineering')

    st.markdown('##### Age')
    age_engineering()


    st.markdown('##### Balance')
    balance_engineering()

    st.markdown('##### Duration')
    duration_engineering()

    st.markdown('##### Campaign')
    campaign_engineering()

    st.markdown('##### Pdays')
    pdays_engineering()

    st.markdown('##### Previous')
    previous_engineering()
    
















    


    
if __name__ == "__main__":
    main()
