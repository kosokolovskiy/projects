import asyncio
from threading import Thread
import threading
from typing import Optional
from openai import OpenAI

from pytube import YouTube
from moviepy.editor import VideoFileClip

from pydub import AudioSegment

import os
from pathlib import Path
import time

import streamlit as st

from projects.project_4.utils.aws.aws_funcs import upload_to_s3, folder_exists_in_s3, check_file_exists_s3, download_from_s3
from projects.project_4.utils.validation_youtube_video import extract_video_id

class File_Proccessor:
    def __init__(self, link, video_name, language):
        self.link = link
        self.video_name = video_name
        self.language = language
        self._transcript = '' 
        self.unique_video_id = extract_video_id(link)

        self.app_name = 'summarizer_app'

        self.extension_video = 'mp4'
        self.extension_audio = 'mp3'
        self.where_to_store_video = f'{self.app_name}/{self.unique_video_id}/videos/{self.video_name}.{self.extension_video}'
        self.where_to_store_audio = f'{self.app_name}/{self.unique_video_id}/audios/{self.video_name}.{self.extension_audio}'
        # st.info(self.where_to_store_audio)
        self.temp_dir = Path('/tmp')
        self.temp_dir.mkdir(exist_ok=True)
        # print(list(self.temp_dir.iterdir()))
        # st.info(list(self.temp_dir.iterdir()))



    @property
    def transcript(self):
        print('INSIDE PROPERTY')
        if self._transcript:
            st.markdown(self._transcript)
            return self._transcript
        st.error('No transcript is available now')

    def video_from_youtube(self):
        try:
            print('1')
            youtube_obj = YouTube(self.link)
            stream = youtube_obj.streams.filter(progressive=True, file_extension=self.extension_video).order_by('resolution').desc().first()

            if folder_exists_in_s3(
                f'{self.app_name}/{self.unique_video_id}'
            ) and check_file_exists_s3(self.where_to_store_video):
                print('Folder or File already in S3')
            elif stream:
                file_path_to_upload = stream.download(output_path='/tmp')
                upload_to_s3(file_path_to_upload, self.where_to_store_video)
            else:
                return None
        except Exception as e:
            st.error(f'Error downloading video: {e}')
            print('ERROR')
            print(f'Error downloading video: {e}')
            return None

    def video_to_audio(self):  # sourcery skip: extract-method

        tmp_dir = self.temp_dir / self.unique_video_id 
        tmp_dir.mkdir(exist_ok=True)

        video_local_path = tmp_dir / f"{self.video_name}.{self.extension_video}"
        audio_local_path = tmp_dir / f"{self.video_name}.{self.extension_audio}"

        try:
            if not video_local_path.exists():
                print(f"Downloading video '{self.where_to_store_video}' from S3 to '{video_local_path}'")
                download_from_s3(self.where_to_store_video, str(video_local_path))

            if not audio_local_path.exists():
                print(f"Extracting audio from video '{video_local_path}'")
                video = VideoFileClip(str(video_local_path))
                video.audio.write_audiofile(str(audio_local_path))
                print(f"Extracted audio to '{audio_local_path}' successfully.")

            if not check_file_exists_s3(f'{self.where_to_store_audio}'):
                print(f"Uploading audio '{audio_local_path}' to S3 at '{self.where_to_store_audio}'")
                upload_to_s3(str(audio_local_path), self.where_to_store_audio)
                print("Audio uploaded successfully.")
            else:
                print('Audio is already in S3')
            
        except Exception as e:
            print(f"Error processing video to audio: {e}")

        
    def segment_and_transcribe_audio(self):
        tmp_dir = self.temp_dir / self.unique_video_id 
        full_audio_local_path = tmp_dir / f"{self.video_name}.{self.extension_audio}"

        try:
            if not full_audio_local_path.exists():
                print(f'Downloading audio from S3 to {full_audio_local_path}')
                download_from_s3(self.where_to_store_audio, str(full_audio_local_path))

            audio = AudioSegment.from_mp3(str(full_audio_local_path))
            part_length = 60 * 1000 * 10  

            for i, part_start in enumerate(range(0, len(audio), part_length), 1):
                part_end = part_start + part_length
                part = audio[part_start:part_end]

                part_filename = tmp_dir / f'audio_{self.video_name}_part_{i}.mp3'
                if not part_filename.exists():
                    part.export(part_filename, format='mp3')

                part_s3_path = f'{self.app_name}/{self.unique_video_id}/audios/parts/audio_{self.video_name}_part_{i}.mp3'
                if not check_file_exists_s3(part_s3_path):
                    upload_to_s3(str(part_filename), part_s3_path)
                else:
                    print(f'{part_s3_path} already exists in S3')

                transcription_local_path = tmp_dir / f'text_{self.video_name}_part_{i}.txt'
                transcription_s3_path = f'{self.app_name}/{self.unique_video_id}/texts/parts/text_{self.video_name}_part_{i}.txt'

                if check_file_exists_s3(transcription_s3_path):
                    print('Transcription already exists in S3')
                    download_from_s3(transcription_s3_path, transcription_local_path)

                elif transcript_text := asyncio.run(
                        self.get_transcript(
                            str(part_filename), f'Transcribing Part {i}'
                        )
                    ):

                    with open(transcription_local_path, 'w') as file:
                        file.write(transcript_text)

                    print(transcript_text)
                    upload_to_s3(str(transcription_local_path), transcription_s3_path)

            print("Segmentation, transcription, and uploading completed.")

        except Exception as e:
            print(f"Error during processing: {e}")

    def combine_transcription_parts(self):
        tmp_dir = self.temp_dir / self.unique_video_id 
        count = 1
        for _, _, files in os.walk(tmp_dir.__str__()):
            print(files)
            for one_file in sorted(files):
                if '.txt' in one_file and 'part' in one_file:
                    path_to_part =  tmp_dir / f'text_{self.video_name}_part_{count}.txt'
                    with open(path_to_part, 'r') as file:
                        temp = file.readline()
                    mode = 'w' if count == 1 else 'a'
                    with open(str(tmp_dir / f'{self.video_name}_full.txt'), mode) as file:
                        file.write(temp)
                        self._transcript += temp
                    count += 1

        transcription_s3_path_full = f'{self.app_name}/{self.unique_video_id}/texts/text_{self.video_name}_full.txt'
        if check_file_exists_s3(transcription_s3_path_full):
            print('Full Transcript is already on S3')
        else:
            upload_to_s3(str(tmp_dir / f'{self.video_name}_full.txt'), transcription_s3_path_full)

    def print_text_while_waiting_for_transcription(self, text, stop_event):
        while not stop_event.is_set():
            print(text)
            time.sleep(1)

    async def get_transcript(self, audio_file_path: str, text_for_waiting_user: str) -> Optional[str]:
        client = OpenAI(api_key=st.secrets['OPENAI_KEY'])
        stop_event = threading.Event()
        transcript = None
        async def transcribe_audio() -> None:
            nonlocal transcript
            try:
                with open(audio_file_path, 'rb') as audio_file:
                    response = client.audio.transcriptions.create(
                        model='whisper-1',
                        file=audio_file,
                        language=self.language
                    )
                    transcript = response.text
            except Exception as e:
                print('ERROR')
            finally:
                stop_event.set()

        draw_thread = Thread(target=self.print_text_while_waiting_for_transcription,
                            args=(text_for_waiting_user, stop_event)
        )
        draw_thread.start()

        await asyncio.create_task(transcribe_audio())

        draw_thread.join()
        return transcript


    def pipeline(self):
        self.video_from_youtube()
        self.video_to_audio()
        self.segment_and_transcribe_audio()
        self.combine_transcription_parts()



def main():
    url = 'https://www.youtube.com/watch?v=S_auwUqRcPI'
    video_name = 'DB_Streik_test_short'
    language = 'de'
    obj = File_Proccessor(url, video_name, language)
    obj.pipeline()


if __name__ == '__main__':
    main()

    
