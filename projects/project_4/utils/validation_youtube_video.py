import requests
import streamlit as st
import re

def extract_video_id(url):
    regex = r'(?:https?:\/\/)?(?:www\.|m\.)?(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))([^\?&"\'<>\s]+)'
    return match[1] if (match := re.search(regex, url)) else None


def convert_YouTube_duration_to_seconds(duration):
    day_time = duration.split('T')
    day_duration = day_time[0].replace('P', '')
    day_list = day_duration.split('D')
    if len(day_list) == 2:
        day = int(day_list[0]) * 60 * 60 * 24
        day_list = day_list[1]
    else:
        day = 0
        day_list = day_list[0]
    hour_list = day_time[1].split('H')
    if len(hour_list) == 2:
        hour = int(hour_list[0]) * 60 * 60
        hour_list = hour_list[1]
    else:
        hour = 0
        hour_list = hour_list[0]
    minute_list = hour_list.split('M')
    if len(minute_list) == 2:
        minute = int(minute_list[0]) * 60
        minute_list = minute_list[1]
    else:
        minute = 0
        minute_list = minute_list[0]
    second_list = minute_list.split('S')
    second = int(second_list[0]) if len(second_list) == 2 else 0
    return day + hour + minute + second
    
def check_youtube_video_exists(user_url):
    if video_id := extract_video_id(user_url):
        params = {
            'part': 'contentDetails', 
            'id': video_id,
            'key': st.secrets['YOUTUBE_KEY']
        }
        url = "https://www.googleapis.com/youtube/v3/videos"
        response = requests.get(url, params=params)
        data = response.json()
        duration = convert_YouTube_duration_to_seconds(data.get('items', [{}])[0].get('contentDetails', {}).get('duration'))

        return len(data.get('items', [])) > 0, duration
    return 0