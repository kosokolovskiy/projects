import streamlit as st

def main():
    st.markdown('#### Final Thoughts')

    st.markdown('''
    **It is impossible to create the model that will be about 100% accuracy:**

    1) Some Emojis in some sense the same and are interchangeable, like *Sob* and *Weary* -> 
    confusion is inevitable 
    2) Some Emojis can be used in the same emotional situations that humans want to express
    3) Sometimes people like to be 'creative' and use 'random' emojis, that also makes the model
    confused
    ''')

    st.markdown('''
        **Ways to improve:**

        1) Handle imbalanced data: 

        - scrape more tweets with the emojis that are lacking in a given dataset
        - undersample the most represented, the *Sob*, category

        2) try to create a model, more precisly to fine-tune, the model from Hugging Face, that are for
            text classification tasks
    ''')