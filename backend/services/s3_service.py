import os
import posixpath

import uuid
import boto3
from io import BytesIO
from botocore.exceptions import ClientError
from backend.core.config import setting
from botocore.client import Config

from utils.logger import logger

s3_client = boto3.client(
    "s3",
    aws_access_key_id=setting.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=setting.AWS_SECRET_ACCESS_KEY,
    region_name=setting.AWS_REGION,
    config=Config(signature_version='s3v4', region_name=setting.AWS_REGION)
)


def upload_image(image_bytes: BytesIO) -> str:
    file_name = f"generated-images/{uuid.uuid4()}.png"

    try:
        s3_client.upload_fileobj(
            image_bytes,
            setting.S3_BUCKET_NAME,
            file_name,
            ExtraArgs={"ContentType": "image/png"}
        )

        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': setting.S3_BUCKET_NAME,
                'Key': file_name
            },
            ExpiresIn=3600
        )

        return presigned_url

    except ClientError as e:
        logger.error(f"Error S3 Service: {e}")
        raise e


def download_checkpoint(s3_prefix_checkpoint: str, local_path: str, exist_ok: bool = True) -> str:
    if os.path.exists(local_path):
        logger.info("Checkpoint already exists locally.")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=exist_ok)
    file_name = os.path.basename(local_path)
    s3_key = posixpath.join(s3_prefix_checkpoint, file_name)

    try:
        s3_client.download_file(
            Bucket=setting.S3_BUCKET_NAME,
            Key=s3_key,
            Filename=local_path
        )
        logger.info("Download checkpoint successful!")
        return local_path

    except ClientError as e:
        logger.error(f"Error S3 Service: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        raise e