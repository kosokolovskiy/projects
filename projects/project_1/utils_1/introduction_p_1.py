import streamlit as st

def main():
    st.markdown('''
        We have a Time Series problem and the task is to predcit *Stock Price*. 

        #### Data

        As the given data the *INFY NS* stocks are taken in the period of 2010 - 2024.  

        #### Goals

        1) to get acquainted with *Time Series*;
        2) to get experience with data preprocessing in TS;
        3) to apply Machine Learning algorithms on TS, including Deep Learning.
    ''')

if __name__ == '__main__':
    main()