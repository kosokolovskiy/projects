import streamlit as st
import pandas as pd
import numpy as np


def link_widget():
    url = st.text_input('Enter YouTube Link')
    
    if url_checker(url):
        return url

    st.error('Provided link is not of YouTube. Please check your link.')
    return 0

def url_checker(url):
    


def main_4():
    st.markdown('''
        This app allows you to provide the link to YouTube video and obtain the transcript for it. You are welcome to to get transcript for the video uo to 10 minutes!
    ''')


