import boto3
import streamlit as st

from datetime import timedelta, datetime

from botocore.exceptions import NoCredentialsError

from datetime import timezone
S3_BUCKET_NAME_PROJECTS = st.secrets['S3_BUCKET_NAME_PROJECTS']
AWS_ACCESS_KEY_PROJECTS = st.secrets['AWS_ACCESS_KEY_PROJECTS']
AWS_SECRET_KEY_PROJECTS = st.secrets['AWS_SECRET_KEY_PROJECTS']
STREAMLIT_S3_BUCKET_NAME_PROJECTS = st.secrets['STREAMLIT_S3_BUCKET_NAME_PROJECTS']
STREAMLIT_AWS_ACCESS_KEY_PROJECTS = st.secrets['STREAMLIT_AWS_ACCESS_KEY_PROJECTS']
STREAMLIT_AWS_SECRET_KEY_PROJECTS = st.secrets['STREAMLIT_AWS_SECRET_KEY_PROJECTS']


def upload_to_s3(file_path, object_path, deletion_time=120):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_PROJECTS, aws_secret_access_key=AWS_SECRET_KEY_PROJECTS)

    deletion_time = datetime.now(timezone.utc) + timedelta(minutes=deletion_time)
    deletion_time_str = deletion_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    try:
        s3_client.upload_file(file_path,
                              S3_BUCKET_NAME_PROJECTS, 
                              object_path,
                              ExtraArgs={'Metadata': {'deletion-time': deletion_time_str}}  
                            )

        print(f'File "{file_path}" has been uploaded to {S3_BUCKET_NAME_PROJECTS}/{object_path}')
    except NoCredentialsError:
        print('Credentials not available')



def folder_exists_in_s3(folder_prefix):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_PROJECTS, aws_secret_access_key=AWS_SECRET_KEY_PROJECTS)
    
    if not folder_prefix.endswith('/'):
        folder_prefix += '/'
    
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME_PROJECTS, Prefix=folder_prefix, MaxKeys=1)
    
    return 'Contents' in response

def check_file_exists_s3(object_name):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_PROJECTS, aws_secret_access_key=AWS_SECRET_KEY_PROJECTS)
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME_PROJECTS, Key=object_name)
        return True  
    except s3_client.exceptions.ClientError as e:
        return False 


def download_from_s3(object_path, file_path):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_PROJECTS, aws_secret_access_key=AWS_SECRET_KEY_PROJECTS)

    try:
        file = s3_client.download_file(S3_BUCKET_NAME_PROJECTS, object_path, file_path)
        print(f'Object {object_path} has been downloaded from {S3_BUCKET_NAME_PROJECTS} to {file_path}')
    except NoCredentialsError:
        print('Credentials not available')
    except Exception as e:
        print(f'An error occurred: {e}')







def get_from_s3(object_path):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_PROJECTS, aws_secret_access_key=AWS_SECRET_KEY_PROJECTS)

    try:
        return s3_client.get_object(Bucket=S3_BUCKET_NAME_PROJECTS, Key=object_path)
    except NoCredentialsError:
        print('Credentials not available')
    except Exception as e:
        print(f'An error occurred: {e}')
    return 0 







def delete_folder_from_s3(folder_path):  # sourcery skip: use-named-expression
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_PROJECTS, aws_secret_access_key=AWS_SECRET_KEY_PROJECTS)

    try:
        # List all objects within the folder
        objects_to_delete = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME_PROJECTS, Prefix=folder_path)
        
        # Extract object keys from the response
        object_keys = [obj['Key'] for obj in objects_to_delete.get('Contents', [])]
        
        # Delete the objects
        if object_keys:
            response = s3_client.delete_objects(
                Bucket=S3_BUCKET_NAME_PROJECTS,
                Delete={
                    'Objects': [{'Key': obj_key} for obj_key in object_keys]
                }
            )
            print(f'Deleted objects: {object_keys}')
        else:
            print(f'No objects found in folder: {folder_path}')
    except NoCredentialsError:
        print('Credentials not available')
    except Exception as e:
        print(f'An error occurred: {e}')







def delete_object_from_s3(object_key):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_PROJECTS, aws_secret_access_key=AWS_SECRET_KEY_PROJECTS)

    try:
        response = s3_client.delete_object(
            Bucket=S3_BUCKET_NAME_PROJECTS,
            Key=object_key
        )
        print(response)
        print(f'Deleted object: {object_key}')
    except NoCredentialsError:
        print('Credentials not available')
    except Exception as e:
        print(f'An error occurred: {e}')