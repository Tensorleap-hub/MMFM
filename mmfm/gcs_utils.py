from functools import lru_cache
from os import environ
import json

from google.oauth2 import service_account
from google.cloud import storage
from google.cloud.storage import Bucket
from typing import Optional, Any
import os

from mmfm.config import cnf


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    auth_secret_string = environ['AUTH_SECRET']
    auth_secret = json.loads(auth_secret_string)
    if type(auth_secret) is dict:
        # getting credentials from dictionary account info
        credentials = service_account.Credentials.from_service_account_info(auth_secret)
    else:
        # getting credentials from path
        credentials = service_account.Credentials.from_service_account_file(auth_secret)
    project = credentials.project_id
    gcs_client = storage.Client(project=project, credentials=credentials)
    return gcs_client.bucket(bucket_name)


def _download(bucket_name: str, cloud_file_path: str, local_file_path: Optional[str] = None, overwrite: bool = False) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        local_file_path = os.path.join(cnf.local_data_path, cloud_file_path)

    # check if file is already exists
    if os.path.exists(local_file_path) and not overwrite:
        return local_file_path

    bucket = _connect_to_gcs_and_return_bucket(bucket_name)
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


# def _upload(local_file, destination_blob_name):
#     """Uploads a file to the GCS bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(cnf['BUCKET_NAME'])
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(local_file)
#
