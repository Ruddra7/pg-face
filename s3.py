import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "vipras")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION", "ap-south-2")

s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)

def ensure_bucket_exists():
    """Creates the S3 bucket if it doesn't already exist."""
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"[S3] Bucket '{S3_BUCKET}' exists and is accessible.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"[S3] Bucket '{S3_BUCKET}' not found. Creating...")
            s3_client.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={'LocationConstraint': S3_REGION}
            )
            print(f"[S3] Bucket '{S3_BUCKET}' created successfully.")
        else:
            print(f"[S3] Bucket check error: {e}")

def upload_to_s3(local_file_path, s3_folder):
    """
    Uploads a file to S3 and returns the public URL.
    Falls back gracefully if upload fails.
    """
    if not local_file_path or not os.path.exists(local_file_path):
        print(f"[S3] File not found: {local_file_path}")
        return None

    filename = os.path.basename(local_file_path)
    s3_path = f"{s3_folder}/{filename}"

    try:
        print(f"[S3] Uploading '{filename}' -> s3://{S3_BUCKET}/{s3_path}")
        s3_client.upload_file(
            local_file_path,
            S3_BUCKET,
            s3_path,
            ExtraArgs={'ContentType': 'image/jpeg'}  # Removed ACL to avoid access denied on new buckets
        )
        url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_path}"
        print(f"[S3] Uploaded successfully: {url}")
        return url
    except NoCredentialsError:
        print("[S3] Error: Credentials not available or invalid.")
        return None
    except ClientError as e:
        print(f"[S3] ClientError: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
        return None
    except Exception as e:
        print(f"[S3] Unexpected Error: {e}")
        return None


# --- Initialization ---
# Ensure bucket exists on startup
try:
    ensure_bucket_exists()
except Exception as e:
    print(f"[S3] Bucket initialization warning: {e}")

if __name__ == "__main__":
    """Quick test to verify bucket access and upload."""
    print("Testing S3 connection...")

    # Create a small test image
    test_path = "s3_test.jpg"
    import cv2
    import numpy as np
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(test_img, "TEST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(test_path, test_img)

    url = upload_to_s3(test_path, "test_uploads")
    if url:
        print(f"\n✅ Test PASSED! Image URL:\n{url}")
    else:
        print("\n❌ Test FAILED. Check the error messages above.")

    os.remove(test_path)