import streamlit as st

def main():
    st.markdown('''
    ### Final Thoughts:

    There were several types of models used in this project:
        
    - Simple Dense
    - Simple Conv1D
    - Dense with additional features
    - Ridge
    - Lasso
    - XGBoost
    - SVG Regressor
    - GradBoostRegressor

    The best results are show in the DL models, but also by *Ridge Regression*, especially with additional features. Other ML models 
    demostrated good train error, but bad test, which means that overfitting occured.

    Using *XGBoost* model we could see the feature importance and it can be concluded that the price of previous day
    and all the *Volumes* in general play the crucial role in our forecast. Additional features that were artificially
    introduced didn't play important role in the process of making predictions. 

    *Ways to improve*:

    - Using statistical models for TS Forecasting, for example, ARIMA model;
    - Probe *LSTM* model and different its variant;
    - Probe transofrmer model from *Hugging Face*;

    
    ''')

if __name__ == '__main__':
    main()