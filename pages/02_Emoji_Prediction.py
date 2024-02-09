import streamlit as st
import pandas as pd
from projects.project_2.utils_2 import introduction_p_2
from projects.project_2.utils_2 import eda_p
from projects.project_2.utils_2 import model_selection_p
from projects.project_2.utils_2 import conclusions_p_2

def eda():
    st.header('EDA')
    eda_p.main()

def introduction():
    st.header('Introduction')
    introduction_p_2.main()

def model_selection():
    st.header('Model Selection')
    model_selection_p.main()


def conclusions():
    st.header('Conclusions')
    conclusions_p_2.main()

FUNC_DICT = {
   'Introduction': introduction,
    'EDA': eda,
    'Model Selection': model_selection,
    'Conclusions': conclusions
}


# Callback function for A and B
def onClick(selection_input):
    if selection_input == 'Conclusions':
        st.session_state['selection_project_2'] = 3
    elif selection_input == 'EDA':
        st.session_state['selection_project_2'] = 1
    elif selection_input == 'Introduction':
        st.session_state['selection_project_2'] = 0
    elif selection_input == 'Model Selection':
        st.session_state['selection_project_2'] = 2

def main_project_2():
    st.title('Project 2')

    if 'selection_project_2' not in st.session_state:
        st.session_state['selection_project_2'] = 0

    type_analysis = st.sidebar.selectbox('Type of Analysis: ', ('Pandas'))


    step_analysis_lst =  ('Introduction', 'EDA', 'Model Selection', 'Conclusions')
    step_analysis = st.sidebar.selectbox('Step of Analysis: ', step_analysis_lst, index=st.session_state['selection_project_2'])


    if type_analysis != None and step_analysis != None:
        place_now = step_analysis_lst.index(step_analysis)

        FUNC_DICT[step_analysis]()

        col1, col2 = st.columns([1, 1])
        with col2:
            if place_now < len(step_analysis_lst) - 1:
                st.write('Next Step: ', step_analysis_lst[place_now + 1])
                next_section_button = st.button('Next Section', on_click=onClick, args=(step_analysis_lst[place_now + 1], ))
        with col1:
            if place_now > 0:
                st.write('Previous Step: ', step_analysis_lst[place_now - 1])
                previous_section_button = st.button('Previous Section', on_click=onClick, args=(step_analysis_lst[place_now - 1], ))

        st.write(f'Current Step of Analysis: {step_analysis}')






if __name__ == '__main__':
    main_project_2()
