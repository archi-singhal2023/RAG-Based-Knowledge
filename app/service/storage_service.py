import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from app.config import Config


class StorageService:
    """
    Optional cloud backup layer.
    Uploads processed documents to S3 for persistence across server restarts.
    Call upload_pdf() after a successful vector store creation in main.py
    if cloud backup is desired.
    """

    def __init__(self):
        # FIX #2: StorageService is now a proper, importable, self-contained service.
        # Wire it into main.py by importing and calling upload_pdf() after indexing.
        if not all([Config.AWS_ACCESS_KEY_ID, Config.AWS_SECRET_ACCESS_KEY,
                    Config.AWS_REGION, Config.BUCKET_NAME]):
            raise EnvironmentError(
                "AWS credentials are not fully configured. "
                "Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, "
                "and BUCKET_NAME in your .env file."
            )

        self.s3 = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION
        )
        self.bucket = Config.BUCKET_NAME

    def upload_pdf(self, file_path: str, object_name: str = None) -> bool:
        """
        Uploads a PDF file to the configured S3 bucket.

        Args:
            file_path: Local path to the file to upload.
            object_name: S3 key name. Defaults to the file's basename.

        Returns:
            True on success, False on failure.
        """
        if not os.path.exists(file_path):
            print(f"❌ StorageService: File not found at '{file_path}'")
            return False

        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3.upload_file(file_path, self.bucket, object_name)
            print(f"✅ StorageService: Uploaded '{object_name}' to s3://{self.bucket}/")
            return True
        except (BotoCoreError, ClientError) as e:
            print(f"❌ StorageService: S3 upload failed — {e}")
            return False

    def list_uploads(self) -> list:
        """Returns a list of all object keys currently in the S3 bucket."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except (BotoCoreError, ClientError) as e:
            print(f"❌ StorageService: Could not list bucket contents — {e}")
            return []