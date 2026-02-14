import boto3
from app.config import Config

class StorageService:
    def __init__(self):
        # Create an S3 client using credentials from Config class
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION
        )

    def upload_file(self, file_path, object_name=None):
        """
        Uploads a file to the S3 bucket defined in Config
        """
        if object_name is None:
            object_name = os.path.basename(file_path)
        try:
            self.s3.upload_file(file_path, Config.BUCKET_NAME, object_name)
            print(f"✅ Uploaded to S3: {object_name}")
            return True
        except Exception as e:
            print(f"❌ S3 Upload Error: {e}")
            return False