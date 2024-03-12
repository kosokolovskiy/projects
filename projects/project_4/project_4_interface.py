import streamlit as st
import pandas as pd
import numpy as np
from projects.project_4.utils.validation_youtube_video import check_youtube_video_exists, extract_video_id
from projects.project_4.utils.file_proccessor import File_Proccessor
from projects.project_4.utils.summarazier import Summarizer
from pathlib import Path


LANGUAGES_D = {
            'English': 'en',
            'Deutsch': 'de'
        }

def process_video(url, video_name, language):
    obj = File_Proccessor(url, video_name, language)
    obj.pipeline()
    return obj

def summarize_video(url, video_name):
    obj = Summarizer(extract_video_id(url), video_name)
    obj.summarizer_open_ai()
    return obj

def link_widget():

    st.session_state.url = st.text_input('Enter YouTube Link')

    try:
        check, duration = check_youtube_video_exists(st.session_state.url)
        if duration > 600:
            st.error('Please choose different video, since this one is longer than 10 minutes')
            return 0
    except Exception:
        check = 0

    if st.session_state.url and check:
        st.success('The Link is Valid, Thank You')
        return st.session_state.url

    elif st.session_state.url == '':
        st.info('Please, Enter the link')

    elif not st.session_state.url:
        st.error('No Link is provided')

    else:
        st.error('Provided link is not of YouTube. Please check your link.')
    return 0

def main_4():
    st.markdown('''
        This app allows you to provide the link to YouTube video and obtain the transcript for it. You are welcome to to get transcript for the video up to 10 minutes!
    ''')



    if url := link_widget():
        video_name = st.text_input('Enter Video Name: ', value=st.session_state.get('video_name', ''))
        st.session_state['video_name'] = video_name  


        if 'language' not in st.session_state:
            st.session_state.language = list(LANGUAGES_D.keys())[0]

        language = st.selectbox(label='Choose the language of video:', options=LANGUAGES_D.keys(), index=None)
        st.session_state.language = language

        try:
            if video_name and language:
                transcription_obj = process_video(url, video_name, language)
                with st.expander('Transcript Text'):
                    st.markdown(transcription_obj.transcript)
                    st.download_button(label="Download Text",
                                    data=transcription_obj.transcript,
                                    file_name=f"transcript_{transcription_obj.video_name}.txt",
                                    mime="text/plain")

                obj_summary = summarize_video(url, video_name)
                with st.expander('Summary'):
                    st.markdown(obj_summary.summary_openai)
                    st.download_button(label="Download Summary",
                                        data=obj_summary.summary_openai,
                                        file_name=f"summary_{obj_summary.video_name}.txt",
                                        mime="text/plain")
        except Exception as e:
            print(e)
            st.error('Please, try later')


if __name__ == '__main__':
    main_4()