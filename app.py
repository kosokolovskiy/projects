import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from utils.aws_funcs import get_from_s3

def display_photo(name):
    try:
        if file := get_from_s3(f'/{name}.png'):
            st.image(file['Body'].read())
    except Exception:
            st.error('Object File is not Found')

def main_app():
    st.title("Introduction")
    display_photo('my_photo')

    st.markdown('''          
            Welcome to my website! 
            
            My name is Konstantin Sokolovskiy. As someone on the journey to becoming a specialist in artificial intelligence, I am deeply fascinated by the power of machine learning and data analytics. 
            My journey began at Bauman Moscow State Technical University, where I studied from 2012 to 2018, focusing on gas turbine engines and unconventional 
            power plants. Realizing my true calling lay elsewhere, I moved to Germany in 2020 to pursue a Master's in Computational Engineering. 
            It was here that I fell in love with AI, fascinated by its capabilities and the hands-on implementation process. I try to evolve in this direction in all possble ways: read articles, write, calculate and implement by hand and so on.
            Since 2014, I've also been tutoring in computer science, mathematics, and physics, successfully preparing students for their final school exams. More on this can be found in the next section.
    ''')

    

if __name__ == '__main__':
    main_app()
