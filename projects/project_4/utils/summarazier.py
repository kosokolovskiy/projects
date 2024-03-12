from pathlib import Path
from openai import OpenAI
import streamlit as st
from projects.project_4.utils.aws.aws_funcs import upload_to_s3, folder_exists_in_s3, check_file_exists_s3, download_from_s3

BUCKET_NAME = st.secrets['S3_BUCKET_NAME_PROJECTS']

class Summarizer:
    def __init__(self, video_id, video_name):
        self.video_id = video_id
        self.video_name = video_name
        self.app_name = 'summarizer_app'

        self.tmp_dir = Path(f'/tmp/{self.video_id}')
        self.local_path_to_transcription = self.tmp_dir / f'{self.video_name}_full.txt'
        self.local_path_to_summary_openai = self.tmp_dir / 'summary_{self.video_name}_openai.txt'

        self.path_in_s3_transcription = Path(f'{self.app_name}/{self.video_id}/texts/text_{self.video_name}_full.txt')
        self.path_in_s3_summary_openai = Path(f'{self.app_name}/{self.video_id}/summary/summary_{self.video_name}_openai.txt')

        self.prompt = lambda x: f'Summarize the following text with only important information left:\n\n\n {x}'


    def summarizer_open_ai(self):
        if self.local_path_to_transcription.exists():
            self._get_from_openai()
        elif check_file_exists_s3(self.path_in_s3_transcription.__str__()):
            download_from_s3(self.path_in_s3_transcription, self.local_path_to_transcription)
            self._get_from_openai()

    def _get_from_openai(self):
        client = OpenAI(api_key=st.secrets['OPENAI_KEY'])
        with open(str(self.local_path_to_transcription), 'rb') as file:
            temp = file.readline()
            response = client.chat.completions.create(
                    model='gpt-3.5-turbo-0125',
                    messages=[
                    {
                        "role": "system",
                        "content": "I am a model who undestands the text very good and can do good summarization",
                    },
                    {
                        'role': 'user', 
                        'content': self.prompt(temp)
                    },

                ],
                    temperature=0.5,
                    max_tokens=150,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
            self.summary_openai = response.choices[0].message.content.strip()
            with open(str(self.local_path_to_summary_openai), 'w') as file:
                file.write(self.summary_openai)

        upload_to_s3(self.local_path_to_summary_openai.__str__(), self.path_in_s3_summary_openai.__str__())
        

if __name__ == '__main__':
    obj = Summarizer('S_auwUqRcPI', 'DB_Streik_test_short')
    obj.summarizer_open_ai()