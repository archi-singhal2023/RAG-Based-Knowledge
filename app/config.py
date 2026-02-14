import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

class Config:
    # AWS Credentials
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    BUCKET_NAME = os.getenv("BUCKET_NAME")

    # Hugging Face Token
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # App Settings
    PORT = 8080