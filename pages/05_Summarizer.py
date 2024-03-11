import streamlit as st
import pandas as pd
import os
from projects.project_4.project_4_interface import main_4

def main():
    st.markdown('## Welcome to Summarizer App')
    main_4()

if __name__ == '__main__':
    main()
