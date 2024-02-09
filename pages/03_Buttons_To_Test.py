import streamlit as st
import pandas as pd

import streamlit as st

# Callback function for A and B
def onClick(selection_input):
    if selection_input == 'A':
        st.session_state['selection'] = 0
    if selection_input == 'B':
        st.session_state['selection'] = 1

# Initialize the session state
if 'selection' not in st.session_state:
    st.session_state['selection'] = 0

# Select box
selected = st.selectbox('Make a selection:', ('A', 'B'), index=st.session_state['selection'])

# Buttons
st.button('A', on_click=onClick, args='A')
st.button('B', on_click=onClick, args='B')

# Conditional display of data visualization
if selected == 'A':
    st.subheader('A')

if selected == 'B':
    st.subheader('B')