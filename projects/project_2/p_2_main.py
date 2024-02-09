import streamlit as st
import pandas as pd
import os

from projects.project_2.code_snippets import *
import projects.project_2.utils_2.eda_p as eda
import projects.project_2.utils_2.model_selection_p as ms


S3_BUCKET_NAME_PROJECTS = os.environ['STREAMLIT_S3_BUCKET_NAME_PROJECTS']
AWS_ACCESS_KEY_PROJECTS = os.environ['STREAMLIT_AWS_ACCESS_KEY_PROJECTS']
AWS_SECRET_KEY_PROJECTS = os.environ['STREAMLIT_AWS_SECRET_KEY_PROJECTS']


def p_2_main():

    eda.main()

    # ms.main()

if __name__ == '__main__':
    p_2_main()


