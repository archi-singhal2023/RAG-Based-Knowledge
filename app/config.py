import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # AWS Credentials (for future StorageService integration)
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    BUCKET_NAME = os.getenv("BUCKET_NAME")

    # Hugging Face Token
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # App Settings
    PORT = int(os.getenv("PORT", 8080))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    @classmethod
    def validate(cls):
        """Validates that all required environment variables are set on startup."""
        missing = []
        if not cls.HUGGINGFACEHUB_API_TOKEN:
            missing.append("HUGGINGFACEHUB_API_TOKEN")
        if missing:
            raise EnvironmentError(
                f"❌ Missing required environment variables: {', '.join(missing)}\n"
                f"Please check your .env file."
            )