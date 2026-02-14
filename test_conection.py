import os
from app.service.storage_service import StorageService
from app.config import Config

def test_aws():
    print("🔍 Testing AWS Connection...")
    try:
        storage = StorageService()
        # This will attempt to list objects in your bucket to verify credentials
        response = storage.s3.list_objects_v2(Bucket=Config.BUCKET_NAME)
        print(f"✅ Connection Successful! Found bucket: {Config.BUCKET_NAME}")
        return True
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return False

def test_huggingface():
    print("\n🔍 Testing Hugging Face Token...")
    token = Config.HUGGINGFACEHUB_API_TOKEN
    if token and len(token) > 10:
        print(f"✅ Token detected (starts with: {token[:8]}...)")
    else:
        print("❌ Hugging Face Token missing or invalid in .env")

if __name__ == "__main__":
    test_aws()
    test_huggingface()