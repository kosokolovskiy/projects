import boto3
from datetime import datetime, timezone
import streamlit as st

S3_BUCKET_NAME_PROJECTS = st.secrets['S3_BUCKET_NAME_PROJECTS']

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    folder_prefix = 'summarizer_app/'

    response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME_PROJECTS, Prefix=folder_prefix)
    if 'Contents' in response:
        for item in response['Contents']:
            key = item['Key']
            obj = s3.head_object(Bucket=S3_BUCKET_NAME_PROJECTS, Key=key)
            if deletion_time_str := obj['Metadata'].get('deletion-time', ''):
                deletion_time = datetime.strptime(deletion_time_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > deletion_time:
                    s3.delete_object(Bucket=S3_BUCKET_NAME_PROJECTS, Key=key)
                    print(f'Deleted {key} from {S3_BUCKET_NAME_PROJECTS}.')