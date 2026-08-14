import os
import glob
import boto3
from botocore.exceptions import ClientError

class CheckpointManager:
    def __init__(self, checkpoint_dir, bucket_name=None, s3_prefix=None, max_to_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.max_to_keep = max_to_keep
        
        self.s3_client = boto3.client('s3') if self.bucket_name else None

    def _get_sorted_local_checkpoints(self):
        search_pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.pt")
        checkpoints = sorted(glob.glob(search_pattern), key=os.path.getmtime)
        return [c for c in checkpoints if "checkpoint_latest.pt" not in c]

    def manage_local(self):
        checkpoints = self._get_sorted_local_checkpoints()
        if not checkpoints:
            return

        while len(checkpoints) > self.max_to_keep:
            oldest = checkpoints.pop(0)
            try:
                if os.path.exists(oldest):
                    os.remove(oldest)
                    print(f"[Local] Deleted: {oldest}")
            except OSError as e:
                print(f"[Local] Error deleting {oldest}: {e}")

    def manage_s3(self, specific_checkpoint_path=None):
        if not self.s3_client:
            print("[S3] Warning: bucket_name is not configured, skipping S3.")
            return

        if specific_checkpoint_path:
            latest_checkpoint = specific_checkpoint_path
        else:
            checkpoints = self._get_sorted_local_checkpoints()
            if not checkpoints:
                print("[S3] No local checkpoints found to upload.")
                return
            latest_checkpoint = checkpoints[-1]

        file_name = os.path.basename(latest_checkpoint)
        s3_key = f"{self.s3_prefix.rstrip('/')}/{file_name}"

        try:
            print(f"[S3] Uploading {file_name} to s3://{self.bucket_name}/{s3_key}...")
            self.s3_client.upload_file(latest_checkpoint, self.bucket_name, s3_key)
            print("[S3] Upload successful!")
        except ClientError as e:
            print(f"[S3] Error during upload: {e}")

        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.s3_prefix)
            if 'Contents' not in response:
                return

            s3_objects = [
                obj for obj in response['Contents'] 
                if "checkpoint_latest.pt" not in obj['Key'] and obj['Key'].endswith('.pt')
            ]
            
            s3_objects.sort(key=lambda x: x['LastModified'])

            while len(s3_objects) > self.max_to_keep:
                oldest_obj = s3_objects.pop(0)
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=oldest_obj['Key'])
                print(f"[S3] Deleted: s3://{self.bucket_name}/{oldest_obj['Key']}")

        except ClientError as e:
            print(f"[S3] Error during cleanup: {e}")

    def manage_all(self):
        self.manage_s3()
        self.manage_local()