import streamlit as st
import pandas as pd
import numpy as np

import os
import pathlib

from projects.project_5.utils.main_widget import main_widget

def main():
    st.title('Welcome to Recommendation System App')

    main_widget()




if __name__ == '__main__':
    main()