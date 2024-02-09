import streamlit as st
import pandas as pd
import os

from projects.project_1.code_snippets import *
import projects.project_1.utils_1.eda as eda
import projects.project_1.utils_1.model_selection as ms


def p_1_main():
    eda.main()

if __name__ == '__main':
    p_1_main()


