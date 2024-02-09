import boto3
import os
import streamlit as st

from botocore.exceptions import NoCredentialsError

# S3_BUCKET_NAME_PROJECTS = os.environ['STREAMLIT_S3_BUCKET_NAME_PROJECTS']
# AWS_ACCESS_KEY_PROJECTS = os.environ['STREAMLIT_AWS_ACCESS_KEY_PROJECTS']
# AWS_SECRET_KEY_PROJECTS = os.environ['STREAMLIT_AWS_SECRET_KEY_PROJECTS']


def upload_to_s3(file_path, object_path):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_PROJECTS, aws_secret_access_key=AWS_SECRET_KEY_PROJECTS)

    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME_PROJECTS, object_path,
                # ExtraArgs={
                #     'ACL': 'public-read',
                #     # 'ContentType': 'application/octet-stream'
                #     'ContentType': 'text/csv'
                # }
        )
        print(f'File {file_path} has been uploaded to {S3_BUCKET_NAME_PROJECTS}/{object_path}')
    except NoCredentialsError:
        print('Credentials not available')


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

if __name__ == '__main__':
    S3_BUCKET_NAME_PROJECTS = os.environ['STREAMLIT_S3_BUCKET_NAME_PROJECTS']
    AWS_ACCESS_KEY_PROJECTS = os.environ['STREAMLIT_AWS_ACCESS_KEY_PROJECTS']
    AWS_SECRET_KEY_PROJECTS = os.environ['STREAMLIT_AWS_SECRET_KEY_PROJECTS']
